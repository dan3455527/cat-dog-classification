import torch
import PIL.Image as im
import numpy as np
import time
import torchvision.transforms as transforms
from model_build.resnet import *

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ResNet(BasicBlock, layers=[1]*8, num_class=2)
model.load_state_dict(torch.load("./models/resnet_self_weight.pt"))
model.to(device=device)

transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),
  transforms.CenterCrop((224, 224)),
])

print(time.localtime())
img = im.open("./data/test_set/test_set/cats/cat.4015.jpg")
img = transform(img).unsqueeze(0)
img = img.to(device)
result = model(img)
result = torch.nn.functional.softmax(result)
print(result)
print(time.localtime())
