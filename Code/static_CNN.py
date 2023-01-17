import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import copy
import random

# return alphabet for specific index
static_alphabet=[
    'A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y'
]

# transform for dataload
train_transform = torchvision.transforms.Compose(
    [
     torchvision.transforms.ToTensor(),
     torchvision.transforms.RandomRotation(5),
     torchvision.transforms.RandomRotation(5),
     torchvision.transforms.Grayscale(),
     torchvision.transforms.Resize((112,112)),
     ])

valid_transform = torchvision.transforms.Compose(
    [
     torchvision.transforms.ToTensor(),
     torchvision.transforms.Grayscale(),
     torchvision.transforms.Resize((112,112)),
     ])


# Dataset dataset
class ImageDataset(Dataset):

    #
    def __init__(self, base_dir_list):
      self.dir_labels = []
      n = 0
      # num of char sampled at each file
      size_each_step=[5,5,5,5,5,5,5,5]
      # in case the data is flipped
      is_mirror = [0,0,0,0,0,0,0,0]

      # save name of each data file
      for img_dir in base_dir_list:
        for i in range(24):
          curr_dir = img_dir+ "/" +static_alphabet[i]
          dir_list = os.listdir(curr_dir)
          dir_size = len(dir_list)
          for j in range(size_each_step[n]):
            ind = random.randint(0, dir_size-1)
            path = curr_dir + "/" + dir_list[ind]
            self.dir_labels.append([path, i, is_mirror[n]])
        n += 1

    def __len__(self):
        return len(self.dir_labels)

    def __getitem__(self, idx):
        # read image from saved data file name
        img_path = self.dir_labels[idx][0]
        label = self.dir_labels[idx][1]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.dir_labels[idx][2] == 1:
          img = cv2.flip(img,1)
        if self.transform:
            img = self.transform(img)
        return img, label

# base directory where data exists
base_dir = '/content/drive/MyDrive/static_hand/data'

# load data to dataloader
dir_list = [base_dir + str(i) for i in range(7)]
training_data1=ImageDataset(dir_list, transform = train_transform)
train_dataloader1 = DataLoader(training_data1, batch_size=8, shuffle=True)
training_data2=ImageDataset(dir_list, transform = train_transform)
train_dataloader2 = DataLoader(training_data2, batch_size=8, shuffle=True)


# model for static alphabet CNN
class static_hand(nn.Module):
    def __init__(self):
        super(static_hand, self).__init__()

        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(16)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(16)
        self.relu5 = nn.ReLU()

        self.flat = nn.Flatten(1, 3)
        self.linear = nn.Linear(16 * 7 * 7, 128)
        self.linear2 = nn.Linear(128, 24)
        self.pool_jump = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        y = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        jump = self.pool_jump(y)
        x = self.pool4(self.relu4(self.bn4(self.conv4(y))))
        x = self.relu5(self.bn5(self.conv5(x))) + jump
        x = self.flat(x)
        x = self.linear(x)
        x = self.linear2(x)
        return x

# typical train algorithm
def train(net, dataloader, loss_kind, optimizer): # Function to train the network
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        img = data[0]
        optimizer.zero_grad()
        outputs = net(img)
        loss = loss_kind(outputs, data[1])
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i%50 == 0 and i> 0:
           tl = running_loss / 50
           print("At step ", i , ", loss:", tl)
           running_loss = 0.0

# model initialization and training
model = static_hand()
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.005, betas=(0.5, 0.99))
train(model, train_dataloader1, loss, optimizer)

# test algorithm
test_dir='/content/drive/MyDrive/static_hand/data1/O'
dir_list = os.listdir(test_dir)
alpha = [0 for _ in range(25)]
length = len(dir_list)
for i in range(int(length)):
  dir = dir_list[i]
  img = cv2.imread(test_dir+'/'+dir, cv2.IMREAD_COLOR)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img = cv2.resize(img, (112, 112))
  img = np.expand_dims(img, axis = 0)
  img = np.expand_dims(img, axis = 0)
  img = torch.from_numpy(img)
  img = img.type('torch.FloatTensor')
  con = model(img)
  con = F.softmax(con)
  ind = torch.argmax(con)
  alpha[ind] += 1
prt = []
for i in range(24):
  prt.append([static_alphabet[i],alpha[i]])
print(prt)