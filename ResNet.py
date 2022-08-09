import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models.resnet import ResNet
from torch.utils.data import Dataset, DataLoader
from utils.utils import train, test, plot_curve
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary

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
  
train_datasets = Trainset(data_path_ls_npy="./data/data_list/train_data_path.npy", labels_ls_npy="./data/data_list/labels_onehot.npy")
test_datasets = Trainset(data_path_ls_npy="./data/data_list/test_data_path.npy", labels_ls_npy="./data/data_list/test_labels_onehot.npy")
train_datasets, val_datasets = torch.utils.data.random_split(train_datasets, [int(len(train_datasets)*0.8), int(len(train_datasets)*0.2)])

train_loader = DataLoader(train_datasets, batch_size=100, shuffle=True)
val_loader = DataLoader(val_datasets, batch_size=100, shuffle=False)
test_loader = DataLoader(test_datasets, batch_size=100, shuffle=False)

class ResNet(nn.Module):
  def __init__(self):
    super(ResNet, self).__init__()
    self.model = models.resnet18(pretrained=True)
    self.model.fc = nn.Linear(512, 2)

  def forward(self, x):
    x = self.model(x)
    return x

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ResNet().to(device)
summary(model, (3, 224, 224))

loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

history = train(
  n_epochs=6,
  optimizer=optimizer,
  model=model,
  device=device,
  loss_func=loss_func,
  train_loader=train_loader,
  verbose=True,
  validation_loader=val_loader
)
plot_curve(history, "save")

test_acc = test(model, test_loader, device)
if test_acc > 0.7:
  torch.save(model, "./models/cat-dog-resnet.pt")