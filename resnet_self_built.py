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
from model_build.resnet import * 


# dataloader configuration
transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),
  transforms.CenterCrop((224, 224))
])
# load data

train_datasets = Trainset(data_path_ls_npy="./data/data_list/train_data_path.npy", labels_ls_npy="./data/data_list/labels_onehot.npy", transform=transform)
test_datasets = Trainset(data_path_ls_npy="./data/data_list/test_data_path.npy", labels_ls_npy="./data/data_list/test_labels_onehot.npy", transform=transform)
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