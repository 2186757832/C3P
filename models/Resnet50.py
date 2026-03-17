class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_c)
        self.conv3 = nn.Conv2d(out_c, out_c*self.expansion, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_c*self.expansion)
        self.relu  = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c*self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c*self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c*self.expansion)
            )

    def forward(self, x):
        out  = self.relu(self.bn1(self.conv1(x)))
        out  = self.relu(self.bn2(self.conv2(out)))
        out  = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return self.relu(out)


class ResNet_CIFAR(nn.Module):
    def __init__(self, block, layers, num_classes=100):
        super().__init__()
        self.in_c = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc    = nn.Linear(512*block.expansion, num_classes)

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