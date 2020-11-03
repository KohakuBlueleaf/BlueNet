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
			Conv({'f_num':64, 'f_size':3, 'pad':1, 'stride':1},batchnorm=True),
			Conv({'f_num':64, 'f_size':3, 'pad':1, 'stride':1},batchnorm=True),
			Pool(2,2,2),
			
			Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1},batchnorm=True),
			Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1},batchnorm=True),
			Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1},batchnorm=True),
			Pool(2,2,2),

			Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1},batchnorm=True),
			Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1},batchnorm=True),
			Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1},batchnorm=True),
			Pool(2,2,2),
			
			Flatten(),
			Dropout(0.35),

			Dense(512),
			Dropout(0.35),
			
			Dense(10),
			SoftmaxWithLoss()
		]

## load train set and test set                    Normalize Flat One-hot Smooth type
(x_train,t_train),(x_test,t_test) = cifar.load_cifar(True, False, True, False, np.float32)

##Initialize the neural network       
net = Net(model)
net.initialize(shape = (3,32,32),
							 af = GELU,
							 opt = Adam,
							 rate = 0.0003,
							 init_mode = 'xaiver',
							 dtype = np.float32)
net.load('./cifar10_example/')

##Print the structure of the network
net.print_size()

##Pre learn
#net.pre_train_for_conv(x_train, 50)

##Set some parameters for training 
batch_size = 30
size_for_test = 100
train_size = x_train.shape[0]
iter_per_epoch = max((train_size//batch_size), 1)

start = time.time()
max_acc = net.accuracy(x_test, t_test, size_for_test)
end = time.time()
print('Test Acc :{}'.format(str(max_acc*100)[:5]))
print('Cost Time:{}'.format(str(end-start)[:5]))

##Input how many epoch You wnat
epoch = int(input('Epoch:'))

##Start Training
print('\n┌───────────────────────────┐  ')
print('│Training start             │  ')
for j in range(epoch):
    start = time.time()
    print("│===========================│  ")
    
    for i in range(iter_per_epoch):
        batch_mask = rc(train_size, batch_size)                 #Random choose data
        
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        
        loss = net.train(x_batch, t_batch)					    #Train&Caculate the loss of the net
        print('│Epoch {:<4}  Loss     :{}│  '.format(j+1,str(loss)[:5]), end='\r', flush=True)
    
    cost = time.time()-start
    test_acc = net.accuracy(x_test, t_test, size_for_test)
    
    if test_acc>max_acc:
        max_acc = test_acc 
        net.save('./cifar10_example/')                                                 #Save the parameters
    
    print("│Epoch {:<4} Test Acc  :{:<5}│  ".format(j+1,str(test_acc*100)[:5]))
    print("│           Cost Time :{:<5}│  ".format(str(cost)[:5]))
    
print('└───────────────────────────┘  ')

net.update('./cifar10_example/weight')
print("Final Accuracy: %2f%%"%(100*net.accuracy(x_test, t_test, size_for_test)))