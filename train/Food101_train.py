import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, datasets
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# 配置
os.makedirs('./model_outputs', exist_ok=True)
os.makedirs('./data', exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TransformedSubset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.indices)


class MergedDataset(Dataset):
    def __init__(self, dataset1, indices1, dataset2, indices2, transform=None):
        self.dataset1 = dataset1
        self.indices1 = indices1
        self.dataset2 = dataset2
        self.indices2 = indices2
        self.transform = transform
        self.len1 = len(indices1)
        self.len2 = len(indices2)

    def __getitem__(self, idx):
        if idx < self.len1:
            x, y = self.dataset1[self.indices1[idx]]
        else:
            x, y = self.dataset2[self.indices2[idx - self.len1]]

        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return self.len1 + self.len2


def get_cifar100_loaders(batch_size=128):
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    print("正在加载CIFAR-100原始数据...")
    full_train = datasets.CIFAR100(root='./data', train=True, download=False, transform=None)
    full_test = datasets.CIFAR100(root='./data', train=False, download=False, transform=None)

    train_indices = []
    test_indices_from_train = []
    test_indices_from_test = []
    train_targets = np.array(full_train.targets)
    test_targets = np.array(full_test.targets)

    for class_id in range(100):
        # 从原训练集中抽取
        class_idx = np.where(train_targets == class_id)[0]
        np.random.shuffle(class_idx)

        train_indices.extend(class_idx[:400])
        test_indices_from_train.extend(class_idx[400:])

        test_class_idx = np.where(test_targets == class_id)[0]
        test_indices_from_test.extend(test_class_idx)

    print(f"训练集: {len(train_indices):,}张 (40,000)")
    print(f"测试集(来自原训练): {len(test_indices_from_train):,}张 (10,000)")
    print(f"测试集(来自原测试): {len(test_indices_from_test):,}张 (10,000)")
    print(f"测试集总计: {len(test_indices_from_train) + len(test_indices_from_test):,}张 (20,000)")
    print(f"比例: 4:2 (2:1)")

    # 创建数据集
    train_set = TransformedSubset(full_train, train_indices, transform=train_transform)
    test_set = MergedDataset(
        full_train, test_indices_from_train,
        full_test, test_indices_from_test,
        transform=test_transform
    )

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader, 100


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.conv3 = nn.Conv2d(out_c, out_c * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_c * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c * self.expansion)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return self.relu(out)


class ResNet_CIFAR(nn.Module):
    def __init__(self, block, layers, num_classes=100):
        super().__init__()
        self.in_c = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride):
        layers = [block(self.in_c, planes, stride)]
        self.in_c = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_c, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def resnet50_cifar(num_classes=100):
    return ResNet_CIFAR(Bottleneck, [3, 4, 6, 3], num_classes)


@torch.no_grad()
def evaluate(model, test_loader):
    model.eval()
    correct, total = 0, 0

    for imgs, labs in test_loader:
        imgs, labs = imgs.to(device), labs.to(device)
        with autocast('cuda'):
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
        correct += (preds == labs).sum().item()
        total += labs.size(0)

    model.train()
    return correct / total


def train(model, train_loader, test_loader, epochs=100):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    scaler = GradScaler('cuda') if device.type == 'cuda' else GradScaler()

    model.train()
    for epoch in range(epochs):
        total_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')

        for imgs, labs in pbar:
            imgs, labs = imgs.to(device), labs.to(device)
            optimizer.zero_grad()

            with autocast('cuda'):
                out = model(imgs)
                loss = criterion(out, labs)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(out, 1)
            correct += (preds == labs).sum().item()
            total += imgs.size(0)
            pbar.set_postfix({'loss': f'{total_loss / total:.4f}',
                              'acc': f'{correct / total:.4f}'})

        scheduler.step()
        train_acc = correct / total
        test_acc = evaluate(model, test_loader)

        print(f'Epoch {epoch + 1} | 训练准确率: {train_acc:.4f} | 测试准确率: {test_acc:.4f}')

    save_path = './model_outputs/resnet50_cifar100_42split_200.pth'
    torch.save(model.state_dict(), save_path)
    print(f'模型已保存至 {save_path}')


@torch.no_grad()
def generate_predictions(model, test_loader):
    model.eval()
    softmax = nn.Softmax(dim=1)
    all_preds, all_probs = [], []

    for imgs, _ in tqdm(test_loader, desc='Predicting'):
        imgs = imgs.to(device)
        with autocast('cuda'):
            outs = model(imgs)
            probs = softmax(outs)
        preds = torch.argmax(probs, dim=1)

        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_preds), np.concatenate(all_probs)


def save_outputs(preds, probs, test_loader, num_classes):
    true_labels = []
    for _, labs in test_loader:
        true_labels.extend(labs.numpy())
    true_labels = np.array(true_labels)

    df = pd.DataFrame({
        'sample_idx': np.arange(len(preds)),
        'true_label': true_labels,
        'pred_label': preds,
        'max_prob': np.max(probs, axis=1)
    })

    prob_cols = [f'prob_cls{i}' for i in range(num_classes)]
    prob_df = pd.DataFrame(probs, columns=prob_cols)
    df = pd.concat([df, prob_df], axis=1)

    output_path = './model_outputs/cifar100_test_predictions_200.csv'
    df.to_csv(output_path, index=False, float_format='%.6f')
    print(f'预测结果已保存至 {output_path}')
    print(f'总计样本数: {len(df)} (应20,000)')


if __name__ == '__main__':
    train_loader, test_loader, num_classes = get_cifar100_loaders(batch_size=128)
    print(f"\n检测到CIFAR-100数据集（4:2划分），共 {num_classes} 个类别")

    model = resnet50_cifar(num_classes=num_classes).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    train(model, train_loader, test_loader, epochs=200)
    preds, probs = generate_predictions(model, test_loader)
    save_outputs(preds, probs, test_loader, num_classes)
    print("\n验证测试集类别分布...")
    labels = []
    for _, labs in test_loader:
        labels.extend(labs.numpy())
    unique, counts = np.unique(labels, return_counts=True)
    print(f"类别范围: {unique.min()} - {unique.max()}")
    print(f"每类样本数: {counts[0]} (应200)")
    print(f"总样本数: {len(labels)} (应20,000)")