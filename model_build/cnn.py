import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
  def __init__(self) -> None:
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(3, 16, 3)
    self.conv2 = nn.Conv2d(16, 32, 3)
    self.conv_bn1 = nn.BatchNorm2d(32)
    self.conv3 = nn.Conv2d(32, 64, 3)
    self.conv4 = nn.Conv2d(64, 128, 3)
    self.conv_bn2 = nn.BatchNorm2d(128)
    self.pool = nn.MaxPool2d(2)
    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(128*9*9, 550)
    self.fc2 = nn.Linear(550, 400)
    self.fc_bn1 = nn.BatchNorm1d(400)
    self.fc3 = nn.Linear(400, 300)
    self.fc4 = nn.Linear(300, 200)
    self.fc_bn2 = nn.BatchNorm1d(200)
    self.fc5 = nn.Linear(200, 2)
    self.drop = nn.Dropout(0.25)
  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv_bn1(self.conv2(x))))
    x = self.pool(F.relu(self.conv3(x)))
    x = self.pool(F.relu(self.conv_bn2(self.conv4(x))))
    x = self.flatten(x)
    x = F.relu(self.fc1(x))
    x = self.drop(x)
    x = F.relu(self.fc_bn1(self.fc2(x)))
    x = self.drop(x)
    x = F.relu(self.fc3(x))
    x = self.drop(x)
    x = F.relu(self.fc_bn2(self.fc4(x)))
    x = self.drop(x)
    x = self.fc5(x)
    x = F.softmax(x, dim=1)
    return x