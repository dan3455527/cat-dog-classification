import torch
import datetime
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


# data related
def image_loader(path, transform):
  image = Image.open(path)
  image_tensor = transform(image)
  return image_tensor

class Trainset(Dataset):
  def __init__(self, data_path_ls_npy, labels_ls_npy, transform, loader=image_loader):
    self.transform = transform
    self.data = np.load(data_path_ls_npy)
    self.target = np.load(labels_ls_npy)
    self.loader = loader

  def __getitem__(self, index):
    fn = self.data[index]
    data = self.loader(fn, transform=self.transform)
    target = self.target[index]
    return data, target
  
  def __len__(self):
    return len(self.data)


def update_lr(optimizer, lr):
  for param_group in optimizer.param_groups:
    param_group["lr"] = lr

def train(n_epochs, optimizer, model, device, loss_func, lr, train_loader, verbose=True, validation_loader=None, lr_update=False):
  history = {"loss":[], "acc":[], "val_loss":[], "val_acc":[]}
  model.to(device)
  curr_lr = lr
  for epoch in range(1, n_epochs+1):
    running_loss = 0.0
    correct = 0.0
    val_loss = 0.0
    val_correct = 0.0
    train_total = 0
    val_total = 0
    
    # training part
    model.train()
    for batch_idx, (input, labels) in enumerate(train_loader):
      input, labels = input.to(device), labels.type(torch.LongTensor)
      labels = labels.to(device)
      optimizer.zero_grad()
      output = model(input)
      loss =  loss_func(output, labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()


    # validation part
    if validation_loader != None:
      model.eval()
      for _batch_idx, (val_input, val_labels) in enumerate(validation_loader):
        val_input, val_labels = val_input.to(device), val_labels.type(torch.LongTensor)
        val_labels = val_labels.to(device)
        val_output = model(val_input)
        loss = loss_func(val_output, val_labels)
        val_loss += loss.item()

    if epoch == 1 or epoch % 1 == 0:
      # training loss and accuracy calculation
      pred = torch.max(output, dim=1)[1]
      correct += (pred == labels).sum().item()
      train_total += labels.size(0)
      acc = correct / train_total
      output_str = f"{datetime.datetime.now()},Epoch {epoch}\tTraining loss {running_loss / len(train_loader):.6f}, Acc {acc:.6f} "
      history["loss"].append(running_loss/ len(train_loader.dataset))
      history["acc"].append(acc)
      # validation loss and accuracy calculation
      if validation_loader != None:
        val_pred = torch.max(val_output, dim=1)[1]
        val_correct += (val_pred == val_labels).sum().item()
        val_total += val_labels.size(0)
        val_acc = val_correct / val_total
        output_str += f"Val loss {val_loss / len(validation_loader):.6f}, val_acc {val_acc:.6f}"
        history["val_loss"].append(val_loss/ len(validation_loader.dataset))
        history["val_acc"].append(val_acc)
      print(output_str, end="")
      if verbose == False:
        print("\r", end="", flush=True)
      else:
        print("\n")
    # decay lr
    if lr_update == True:
      if (epoch + 1) % 8 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)
  return history

def test(model, test_loader, device):
  model.eval()
  model.to(device)
  test_correct = 0.0
  test_total = 0
  with torch.no_grad():
    for _batch_id, (input, labels) in enumerate(test_loader):
      input, labels = input.to(device), labels.type(torch.LongTensor)
      labels = labels.to(device)
      output = model(input)
      pred = torch.max(output, dim=1)[1]
      test_total += labels.size(0)
      test_correct += (pred == labels).sum().item()
      test_acc = test_correct / test_total
    print(f"Accuracy: {test_acc:.6f}")
  return test_acc

def plot_curve(history_dict, mode="show", prefix=""):
  if prefix != "":
    prefix = prefix + "_"
  plt.plot(history_dict["loss"])
  if history_dict["val_loss"]:
    plt.plot(history_dict["val_loss"])
  plt.title("model loss")
  plt.ylabel("loss")
  # plt.ylim(0,)
  plt.xlabel("epochs")
  plt.legend(["train", "val"], loc="upper right")
  if mode == "show":
    plt.show()
  elif mode == "save":
    plt.savefig(f"./{prefix}model_loss.png")
  
  plt.clf()

  plt.plot(history_dict["acc"])
  if history_dict["val_acc"]:
    plt.plot(history_dict["val_acc"])
  plt.title("model acc")
  plt.ylabel("acc")
  plt.ylim(0,)
  plt.xlabel("epochs")
  plt.legend(["train", "val"], loc="upper right")
  plt.tight_layout()
  if mode == "show":
    plt.show()
  elif mode == "save":
    plt.savefig(f"./{prefix}model_accuracy.png")
  pass
