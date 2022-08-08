import os
import torch
import numpy as np
import torch.nn.functional as F
# data storage structure
# test
#   |--cats
#   |--dogs
# train
#   |--cats
#   |--dogs


def data_loader_list():
  data_ls = []
  pass


data_ls = []
label_ls = []
test_data_ls = []
test_label_ls = []

label_class = {"dog":0, "cat":1}
# training set
for root, dirs, files in os.walk("./data/training_set/training_set/cats"):
  for file in files:
    if ".jpg" in file:
      data_ls.append(os.path.join(root, file))
      label_ls.append(0)

for root, dirs, files in os.walk("./data/training_set/training_set/dogs"):
  for file in files:
    if ".jpg" in file:
      data_ls.append(os.path.join(root, file))
      label_ls.append(1)

# test set
for root, dirs, files in os.walk("./data/test_set/test_set/cats"):
  for file in files:
    if ".jpg" in file:
      test_data_ls.append(os.path.join(root, file))
      test_label_ls.append(0)

for root, dirs, files in os.walk("./data/test_set/test_set/dogs"):
  for file in files:
    if ".jpg" in file:
      test_data_ls.append(os.path.join(root, file))
      test_label_ls.append(1) 
      
def onehot(input_list):
  tensor_ls = torch.tensor(input_list)
  tensor_ls = F.one_hot(tensor_ls)
  input_list = tensor_ls.tolist()
  return input_list

label_ls = onehot(label_ls)
test_label_ls = onehot(test_label_ls)

np.save("train_data_path.npy", data_ls)
np.save("labels_onehot.npy", label_ls)
np.save("test_data_path.npy", test_data_ls)
np.save("test_labels_onehot.npy", test_label_ls)