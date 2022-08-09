import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader 
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchsummary import summary
from utils.utils import *

def conv3x3(in_channels, out_channels, stride=1):
  return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
  def __init__(self, in_channels, out_channels, stride=1, downsample=None):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(in_channels, out_channels, stride)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(out_channels, out_channels)
    self.bn2 = nn.BatchNorm2d(out_channels)
    self.downsample = downsample

  def forward(self, x):
    residual = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    if self.downsample:
      residual = self.downsample(x)
    out += residual
    out = self.relu(out)
    return out

class ResNet(nn.Module):
  def __init__(self, block, layers, num_class=2) -> None:
    super(ResNet, self).__init__()
    self.in_channels = 8
    self.conv = conv3x3(3, 8)
    self.bn = nn.BatchNorm2d(8)
    self.relu = nn.ReLU(inplace=True)
    self.adaptmaxpool0 = nn.AdaptiveAvgPool2d((128, 128))
    # residual start
    self.layer0 = self.make_layer(block, 8, layers[0], 1)
    self.layer1 = self.make_layer(block, 8, layers[1], 1)
    self.layer2 = self.make_layer(block, 16, layers[2], 2)
    self.layer3 = self.make_layer(block, 16, layers[3], 1)
    self.layer4 = self.make_layer(block, 32, layers[4], 2)
    self.layer5 = self.make_layer(block, 32, layers[5], 1)
    self.layer6 = self.make_layer(block, 64, layers[6], 2)
    self.layer7 = self.make_layer(block, 64, layers[7], 1)
    self.adaptmaxpool1 = nn.AdaptiveAvgPool2d((1, 1))
    self.flatten = nn.Flatten()
    self.fc0 = nn.Linear(64, 32)
    self.fc1 = nn.Linear(32, num_class)


  def make_layer(self, block, out_channels, blocks, stride=1):
    downsample = None
    if (stride != 1) or (self.in_channels != out_channels):
      downsample = nn.Sequential(
        conv3x3(self.in_channels, out_channels, stride=stride),
        nn.BatchNorm2d(out_channels),
      )
    layers = []
    layers.append(block(self.in_channels, out_channels, stride, downsample))
    self.in_channels = out_channels
    for _i in range(1, blocks):
      layers.append(block(out_channels, out_channels))
    return nn.Sequential(*layers)

  def forward(self, x):
    out = self.conv(x)
    out = self.bn(out)
    out = self.relu(out)
    out = self.adaptmaxpool0(out)
    out = self.layer0(out)
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = self.layer5(out)
    out = self.layer6(out)
    out = self.layer7(out)
    out = self.adaptmaxpool1(out)
    out = self.flatten(out)
    out = self.fc0(out)
    out = self.relu(out)
    out = self.fc1(out)
    out = self.relu(out)
    return(out)


# dataloader configuration
transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),
  transforms.CenterCrop((224, 224))
])

def image_loader(path):
  image = Image.open(path)
  image_tensor = transform(image)
  return image_tensor

class Trainset(Dataset):
  def __init__(self, data_path_ls_npy, labels_ls_npy, loader=image_loader):
    self.data = np.load(data_path_ls_npy)
    self.target = np.load(labels_ls_npy)
    self.loader = loader

  def __getitem__(self, index):
    fn = self.data[index]
    data = self.loader(fn)
    target = self.target[index]
    return data, target
  
  def __len__(self):
    return len(self.data)

# load data

train_datasets = Trainset(data_path_ls_npy="./data/data_list/train_data_path.npy", labels_ls_npy="./data/data_list/labels_onehot.npy")
test_datasets = Trainset(data_path_ls_npy="./data/data_list/test_data_path.npy", labels_ls_npy="./data/data_list/test_labels_onehot.npy")
train_datasets, val_datasets = torch.utils.data.random_split(train_datasets, [int(len(train_datasets)*0.8), int(len(train_datasets)*0.2)])

batch_size = 64
train_loader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_datasets, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_datasets, batch_size=batch_size, shuffle=False)

# main 

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ResNet(BasicBlock, [1, 1, 1, 1, 1, 1, 1, 1], 2).to(device=device)
print(device)
summary(model=model, input_size=(3, 224, 224))
lr = 0.0001
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

hist = train(
  n_epochs=100,
  optimizer=optimizer,
  model=model,
  device=device,
  loss_func=criterion,
  lr=lr,
  train_loader=train_loader,
  validation_loader=val_loader,
  lr_update=True
)

plot_curve(hist, mode="save", prefix="Essay")

test(model=model, test_loader=test_loader, device=device)