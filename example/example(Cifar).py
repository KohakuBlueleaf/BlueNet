import sys,os
import time
import numpy as np
from numpy.random import choice as rc


## Import BlueNet
import bluenet.dataset.Cifar as cifar
from bluenet.network import Net
from bluenet.layer import *
from bluenet.activation import GELU
from bluenet.optimizer import Adam


## VGG16
model = [
  Conv({'f_num':6,'f_size':3,'pad':0,'stride':1}),
  Pool(2,2,2),
  Conv({'f_num':16,'f_size':3,'pad':0,'stride':1}),
  Pool(2,2,2),
  Conv({'f_num':120,'f_size':3,'pad':0,'stride':1}),
  Pool(2,2,2),
  Flatten(),
  Dense(10),
  SoftmaxWithLoss(),
]

## load train set and test set                    Normalize Flat One-hot Smooth type
(x_train,t_train),(x_test,t_test) = cifar.load_cifar(True, False, True, False, np.float32)

##Initialize the neural network       
net = Net(model)
net.initialize(
  shape = (3,32,32),
  af = GELU,
  opt = Adam,
  rate = 0.0003,
  init_mode = 'xaiver',
  dtype = np.float32
)
path = './cifar10_example/'
net.load(path)

##Print the structure of the network
net.print_size()

##Pre learn
#mask = rc(x_train.shape[0],30000)
#net.pre_train_for_conv(x_train[mask], 30)

##Set some parameters for training 
batch_size = 30
train_size = x_train.shape[0]
iter_per_epoch = max((train_size//batch_size), 1)

start = time.time()
max_acc = net.accuracy(x_test, t_test, batch_size)
end = time.time()
print('Test Acc :{}%'.format(str(max_acc*100)[:5]))
print('Cost Time:{}s'.format(str(end-start)[:5]))


##Input how many epoch You wnat
epoch = int(input('Epoch:'))

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
    if i%10==0:
      print('│Iters {:<6} Loss        : {:<8} │  '.format(i,str(loss)[:8]), end='\r', flush=True)
  
  cost = time.time()-start
  test_acc = net.accuracy(x_test, t_test, batch_size)
  
  if test_acc>max_acc:
    max_acc = test_acc 
    net.save(path)                        #Save the parameters
  
  print("│Epoch {:<5} Average Loss : {:<8} │  ".format(j+1, str(loss_t/iter_per_epoch)[:8]))
  print("│            Test Acc     : {:<8} │  ".format(str(test_acc*100)[:8]))
  print("│            Cost Time    : {:<8} │  ".format(str(cost)[:8]))
  print("│            Iters/sec    : {:<8} │  ".format(str(iter_per_epoch/cost)[:8]))
    
print('└────────────────────────────────────┘  ')

net.update(path+'weight')
print("Final Accuracy: %2f%%"%(100*net.accuracy(x_test, t_test)))