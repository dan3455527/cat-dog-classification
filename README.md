# Dog-Cat classification

## Introduction
This is the dog-cat classification using pytorch. current network can only got 71.8% accuracy on test datasets.

## Method
### CNN
self build CNN model.
### ResNet18
pytorch pre-trained resnet18 model.
### Summary
|model|time-cost|accuracy|
|-----|---------|--------|
|CNN|6 min|72.45%|
|ResNet18|4 min|98.12%|
### training curve
|model|accuracy curve|loss curve|
|-----|--------------|----------|
|resnet18|![image](https://github.com/dan3455527/cat-dog-classification/blob/main/model_accuracy_resnet18.png)|![image](https://github.com/dan3455527/cat-dog-classification/blob/main/model_loss_resent18.png)|

## Acknoledgment
datasets:[kaggle data](https://www.kaggle.com/datasets/tongpython/cat-and-dog?resource=download)

