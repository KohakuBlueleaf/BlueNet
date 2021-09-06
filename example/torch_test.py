from bluenet.network import Net
from bluenet.layer import *
from bluenet.activation import *

import torch
nn = torch.nn
af = torch.nn.modules.activation
op = torch.optim



model =[
  Conv({'f_num':32,'f_size':3,'pad':1,'stride':1},batchnorm=True),
  Conv({'f_num':32,'f_size':3,'pad':1,'stride':1},batchnorm=True),
  Pool(2,2,2),
  Dropout(0.35),

  Conv({'f_num':64,'f_size':3,'pad':1,'stride':1},batchnorm=True),
  Conv({'f_num':64,'f_size':3,'pad':1,'stride':1},batchnorm=True),
  Pool(2,2,2),
  Dropout(0.35),

  Conv({'f_num':128,'f_size':3,'pad':1,'stride':1},batchnorm=True),
  Conv({'f_num':128,'f_size':3,'pad':1,'stride':1},batchnorm=True),
  Pool(2,2,2),
  Dropout(0.35),

  Conv({'f_num':256,'f_size':3,'pad':1,'stride':1},batchnorm=True),
  Conv({'f_num':256,'f_size':3,'pad':1,'stride':1},batchnorm=True),
  Pool(2,2,2),
  Dropout(0.35),

  Flatten(),
  Dense(512),
  Dropout(0.35),
  Dense(10,af=ID),
]
net = Net(model)
net.initialize(shape = (3,32,32), af=GELU)
net_torch = net.to_torch()
a = str(net_torch)


class Model(torch.nn.Module):
  def __init__(self):
    super(Model, self).__init__()

    self.seq0 = nn.Sequential(
      nn.Conv2d(3, 32, 3, 1, 1),
      nn.BatchNorm2d(32),
      af.GELU(),
      nn.Conv2d(32, 32, 3, 1, 1),
      nn.BatchNorm2d(32),
      af.GELU(),
      nn.MaxPool2d(2,2),
      nn.Dropout(0.35),
    )

    self.seq1 = nn.Sequential(
      nn.Conv2d(32, 64, 3, 1, 1),
      nn.BatchNorm2d(64),
      af.GELU(),
      nn.Conv2d(64, 64, 3, 1, 1),
      nn.BatchNorm2d(64),
      af.GELU(),
      nn.MaxPool2d(2,2),
      nn.Dropout(0.35),
    )

    self.seq2 = nn.Sequential(
      nn.Conv2d(64, 128, 3, 1, 1),
      nn.BatchNorm2d(128),
      af.GELU(),
      nn.Conv2d(128, 128, 3, 1, 1),
      nn.BatchNorm2d(128),
      af.GELU(),
      nn.MaxPool2d(2,2),
      nn.Dropout(0.35),
    )

    self.seq3 = nn.Sequential(
      nn.Conv2d(128, 256, 3, 1, 1),
      nn.BatchNorm2d(256),
      af.GELU(),
      nn.Conv2d(256, 256, 3, 1, 1),
      nn.BatchNorm2d(256),
      af.GELU(),
      nn.MaxPool2d(2,2),
      nn.Dropout(0.35),
    )

    self.seq4 = nn.Sequential(
      nn.Flatten(),
      nn.Linear(1024,512),
      nn.GELU(),
      nn.Dropout(0.35),
      nn.Linear(512,10),
    )
    
  def forward(self, x):
    x = self.seq0(x)
    x = self.seq1(x)
    x = self.seq2(x)
    x = self.seq3(x)
    x = self.seq4(x)
    return x
     
net = Model()
b = str(net)


print(a==b)