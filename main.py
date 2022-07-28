import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
from utils.utils import *
from utils.email import *

transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),
  transforms.CenterCrop((180, 180))
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
# shape: [batch, 3, 224, 224]

# model
class CNN(nn.Module):
  def __init__(self) -> None:
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(3, 16, 3)
    self.conv2 = nn.Conv2d(16, 32, 3)
    self.conv3 = nn.Conv2d(32, 64, 3)
    self.conv4 = nn.Conv2d(64, 128, 3)
    self.pool = nn.MaxPool2d(2)
    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(128*9*9, 550)
    self.fc2 = nn.Linear(550, 400)
    self.fc3 = nn.Linear(400, 300)
    self.fc4 = nn.Linear(300, 200)
    self.fc5 = nn.Linear(200, 2)
    self.drop = nn.Dropout(0.25)
  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = self.pool(F.relu(self.conv3(x)))
    x = self.pool(F.relu(self.conv4(x)))
    x = self.flatten(x)
    x = F.relu(self.fc1(x))
    x = self.drop(x)
    x = F.relu(self.fc2(x))
    x = self.drop(x)
    x = F.relu(self.fc3(x))
    x = self.drop(x)
    x = F.relu(self.fc4(x))
    x = self.drop(x)
    x = self.fc5(x)
    x = F.softmax(x, dim=1)
    return x


model = CNN()
device = "mps" if torch.backends.mps.is_available() else "cpu"
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
email_notification()

plot_curve(history)

test_accuracy = test(model=model, test_loader=test_loader, device=device)

if test_accuracy >= 0.7:
  torch.save(model, "./models/cat-dog.pt")