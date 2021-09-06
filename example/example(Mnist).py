import sys,os
import time
import numpy as np
from numpy.random import choice as rc

## Import BlueNet
import bluenet.setting
bluenet.setting.change_device('GPU')
import bluenet.dataset.Mnist as mnist
from bluenet.network import Net
from bluenet.layer import *
from bluenet.activation import GELU
from bluenet.optimizer import Adam

model =  [
  Conv({'f_num':8,'f_size':3,'pad':0,'stride':1}),
  Pool(2,2,2),
  Conv({'f_num':10,'f_size':3,'pad':0,'stride':1}),
  Pool(2,2,2),
  Conv({'f_num':105,'f_size':3,'pad':0,'stride':1}),
  Flatten(),
  Dense(10),
  SoftmaxWithLoss(),
]

## load train set and test set                    Normalize Flat One-hot Smooth type
(x_train,t_train),(x_test,t_test) = mnist.load_mnist(True, False, True, True, np.float32)

##Initialize the neural network(Use LeNet)     
net = Net(model)
net.initialize(
  shape = (1,28,28), 
  af = GELU, 
  opt = Adam, 
  rate = 0.001, 
  init_mode = 'xaiver', 
  dtype = np.float32
)

##Print the structure of the network
net.print_size()

##Pre learn
#mask = rc(x_train.shape[0],30000)
#net.pre_train_for_conv(x_train[mask], 30)

##Set some parameters for training 
batch_size = 50
train_size = x_train.shape[0]
iter_per_epoch = max((train_size//batch_size), 1)

##Input how many epoch You wnat
epoch = 3

##Start Training
print('\n┌────────────────────────────────────┐  ')
print('│Training start                      │  ')
for j in range(epoch):
  start = time.time()
  print("│====================================│  ")
  
  loss_t = 0
  for i in range(1,iter_per_epoch+1):
    batch_mask = rc(train_size, batch_size)                 #Random choose data

    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    loss = net.train(x_batch, t_batch)              #Train&Caculate the loss of the net
    loss_t += loss
    if i%50==0:
      print('│Iters {:<6} Loss        : {:<8} │  '.format(i,str(loss)[:8]), end='\r', flush=True)
  
  cost = time.time()-start
  
  print("│Epoch {:<5} Average Loss : {:<8} │  ".format(j+1, str(loss_t/iter_per_epoch)[:8]))
  print("│            Cost Time    : {:<8} │  ".format(str(cost)[:8]))
  print("│            Iters/sec    : {:<8} │  ".format(str(iter_per_epoch/cost)[:8]))
    
print('└────────────────────────────────────┘  ')