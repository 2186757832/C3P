# C3P
This repository contains a implementation of C3P corresponding to the follow paper:C3P: Confusion-Guided Conformal Prediction via Adaptive Support Set Patching
## Overview

Traditional conformal prediction (CP) provides strong coverage guarantees, but in practical applications, it can generate excessively large prediction sets.In long tail distribution datasets, coverage can also exhibit instability. Furthermore, existing algorithms suffer from low efficiency in terms of time complexity. To address these limitations, we introduce Case-based Confusion-Compensated Conformal Prediction (C3P), a novel framework that integrates conformal risk control with dynamic CBR retrieval and adaptation. 
C3P treats the base model’s confusion matrix as a lightweight historical case repository. It aggregates ground-truth cases by predicted label to construct compact, high-probability support sets that actively filter the retrieval space, while simultaneously auditing the unexpected exclusion frequency of each true class to track consumed conformal error budgets during revision. When a class’s filtering error exceeds the safety threshold, an adaptive greedy case-patching strategy injects the vulnerable target case into the support sets of its most frequent historical confusors; this targeted compensation continues until strict budget compliance is achieved. 
In theory, C3P provides class-conditional coverage guarantees that are independent of the underlying classifier and agnostic to data distribution. Computationally, by restricting conformal evaluations to the dynamically adapted support sets, C3P dramatically reduces per-prediction post-processing overhead and accelerates real-time inference in CBR systems without sacrificing reliability. 
Our algorithm reduces the time complexity by one order of magnitude compared with the state-of-the-art methods.

## Contents
The major content of our repo are:
 - `data/` A folder that contains the datasets used in our experiments CIFAR10, CIFAR100, Food101, ImageNet, iNaturalist.
 - `models/` The classifier we used in the experiment.
 - `Results/` A folder that contains different files from different experiments.
 - `test/` Code containing all analysis experimental results.
 - `train/` Contains all training codes.

test folder contains:

1. `Stand.py`: The main code used to run the primary benchmark result experiments.
2. `C3P.py`: The main code used to run the primary experiments of our results.

## Prerequisites

Prerequisites for running our code:
 - numpy
 - pandas
 - pathlib
 - torch
 - typing
 - pickle
 - time
 - sklearn
 - torch
 
