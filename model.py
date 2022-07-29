# Contains the neural network
# Task:- Implement the Neural Network module according to problem statement specifications


from torch import nn
import torch
import torch.nn.functional as F
import numpy as np


class Net(nn.Module):
    def __init__(self, num_channels=3):
        super(Net, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1,3,5)
        self.conv2 = nn.Conv2d(3,5,4)
        self.conv3 = nn.Conv2d(5,3,4)
        self.conv4 = nn.ConvTranspose2d(3,1,5,2)

    def forward(self, x):
        #Inputs - 17x17 patches
        (N,n,m) = x.shape
        y = x.view(N,1,n,m)
        y = torch.tensor(y,dtype=torch.float)    #Otherwise were running into error "expected scalar type Byte but found Float"
        y = self.relu(self.conv1(y))
        y = self.relu(self.conv2(y))
        y = self.relu(self.conv3(y))
        y = self.relu(self.conv4(y))

        return y

model = Net()

