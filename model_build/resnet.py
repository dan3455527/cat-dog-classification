import torch.nn as nn
import torch.nn.functional as Fo

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