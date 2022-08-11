import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
from utils.utils import *
from model_build.cnn import CNN

transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),
  transforms.CenterCrop((180, 180))
])

train_datasets = Trainset(data_path_ls_npy="./data/data_list/train_data_path.npy", labels_ls_npy="./data/data_list/labels_onehot.npy")
test_datasets = Trainset(data_path_ls_npy="./data/data_list/test_data_path.npy", labels_ls_npy="./data/data_list/test_labels_onehot.npy")
train_datasets, val_datasets = torch.utils.data.random_split(train_datasets, [int(len(train_datasets)*0.8), int(len(train_datasets)*0.2)])

train_loader = DataLoader(train_datasets, batch_size=100, shuffle=True)
val_loader = DataLoader(val_datasets, batch_size=100, shuffle=False)
test_loader = DataLoader(test_datasets, batch_size=100, shuffle=False)
# shape: [batch, 3, 224, 224]

model = CNN()
# device = "mps" if torch.backends.mps.is_available() else "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(model)
print(f"GPU state: {device}")

# criterion = nn.NLLLoss()
criterion = nn.CrossEntropyLoss()
lr = 0.0015
optimizer = optim.Adam(model.parameters(), lr=lr)
# optimizer = optim.Adadelta(model.parameters(), lr=lr)
# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
n_epoch = 15

history = train(
  n_epochs=n_epoch,
  optimizer=optimizer,
  model=model,
  device=device,
  loss_func=criterion,
  verbose=True,
  train_loader=train_loader,
  validation_loader=val_loader
)

# plot_curve(history)

test_accuracy = test(model=model, test_loader=test_loader, device=device)