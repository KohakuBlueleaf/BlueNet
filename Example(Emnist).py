import sys,os
import time
import numpy as np
from numpy.random import choice as rc


## Import BlueNet
import BlueNet.Dataset.Emnist as emnist
from BlueNet.Network import Net
from BlueNet.Layer import *
from BlueNet.Activation import GELU
from BlueNet.Optimizer import Adam


## LeNet (big version)
model = [
		Conv({'f_num':12, 'f_size':5, 'pad':2, 'stride':1}),
		Pool(2,2,2),
		Conv({'f_num':32, 'f_size':5, 'pad':0, 'stride':1}),
		Pool(2,2,2),
		Conv({'f_num':240, 'f_size':5, 'pad':0, 'stride':1}),
		Flatten(),
		Dense(output_size=168),
		Dense(output_size=26),
			
		SoftmaxWithLoss()
		]

## load train set and test set                    Normalize Flat One-hot Smooth type
(x_train,t_train),(x_test,t_test) = emnist.load_emnist(True, False, True, False, choose=0, type=np.float32)

##Initialize the neural network     
net = Net(model, (1,28,28), GELU, Adam, 0.001, 0, 'xaiver', np.float32)
net.update('./Emnist_sample_weight/')

##Print the structure of the network
net.print_size()

##Pre learn
mask = rc(x_train.shape[0],30000)
net.pre_train_for_conv(x_train[mask], 50)

##Set some parameters for training 
batch_size = 50
train_size = x_train.shape[0]
iter_per_epoch = max((train_size//batch_size), 1)
max_acc = net.accuracy(x_test, t_test, batch_size)
print('Test Acc:{:5.5}'.format(str(max_acc*100)))

##Input how many epoch You wnat
Epoch = int(input('Epoch:'))

##Start Training
print('\n┌────────────────────────────┐  ')
print('│ Training start             │  ')
for j in range(Epoch):
    start = time.time()
    print("│ ===========================│  ")
    
    for i in range(iter_per_epoch):
        batch_mask = rc(train_size, batch_size)                 #Random choose data
        
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        
        loss = net.train(x_batch, t_batch)					    #Train&Caculate the loss of the net
        print('│ Epoch {:<4}  Loss     :{}│  '.format(j+1,str(loss)[:5]), end='\r', flush=True)
    
    cost = time.time()-start
    test_acc = net.accuracy(x_test, t_test, batch_size)
    train_acc = net.accuracy(x_train, t_train, batch_size)
    
    if test_acc>max_acc:
        max_acc = test_acc 
        net.save('./Emnist_sample_weight/')                                                 #Save the parameters
    
    print("│ Epoch {:<4}  Test Acc :{:<5}│  ".format(j+1,str(test_acc*100)[:5]))
    print("│             Train Acc:{:<5}│  ".format(str(train_acc*100)[:5]))
    print("│             Cost Time:{:<5}│  ".format(str(cost)[:5]))
    
print('└────────────────────────────┘  ')

net.update('./Emnist_sample_weight/')
print("Final Accuracy: %2f%%"%(100*net.accuracy(x_test, t_test)))