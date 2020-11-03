import sys,os
import time
import numpy as np
from numpy.random import choice as rc


## Import BlueNet
import bluenet.dataset.Mnist as mnist
from bluenet.network import Net
from bluenet.layer import *
from bluenet.activation import GELU
from bluenet.optimizer import Adam

model = [
        Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1},batchnorm=True),
        Dropout(0.1),

        Conv({'f_num':8, 'f_size':1, 'pad':1, 'stride':1}),
        Conv({'f_num':16, 'f_size':3, 'pad':1, 'stride':1},batchnorm=True),
        Dropout(0.1),
        Pool(2,2,2),

        Conv({'f_num':16, 'f_size':3, 'pad':1, 'stride':1},batchnorm=True),
        Dropout(0.1),
        Conv({'f_num':32, 'f_size':3, 'pad':1, 'stride':1},batchnorm=True),
        Dropout(0.1),
        Pool(2,2,2),

        Conv({'f_num':16, 'f_size':1, 'pad':1, 'stride':1}),
        Conv({'f_num':16, 'f_size':3, 'pad':1, 'stride':1},batchnorm=True),
        Dropout(0.1),
        
        Conv({'f_num':32, 'f_size':3, 'pad':1, 'stride':1},batchnorm=True),
        Dropout(0.1),
        
        Conv({'f_num':10, 'f_size':1, 'pad':1, 'stride':1}),
        PoolAvg(11,11,1),
        Flatten(),
        SoftmaxWithLoss()
		]

model = [
        Dense(580),
        Dense(10),
        SoftmaxWithLoss()
]

## load train set and test set                    Normalize Flat One-hot Smooth type
(x_train,t_train),(x_test,t_test) = mnist.load_mnist(True, True, True, True, np.float64)

##Initialize the neural network(Use LeNet)     
net = Net(model)
net.initialize(shape=(1,28,28), af=GELU, opt=Adam, rate=0.001, init_mode='kaiming', dtype=np.float64)
net.load('./mnist_sample/')

##Print the structure of the network
net.print_size()

##Pre learn
#mask = rc(x_train.shape[0],30000)
#net.pre_train_for_conv(x_train[mask], 30)

##Set some parameters for training 
batch_size = 50
train_size = x_train.shape[0]
iter_per_epoch = max((train_size//batch_size), 1)

start = time.time()
max_acc = net.accuracy(x_test, t_test, batch_size)
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
        if i%50==0:
            print('│Epoch {:<4}  Loss     :{}│  '.format(j+1,str(loss)[:5]), end='\r', flush=True)
    
    cost = time.time()-start
    test_acc = net.accuracy(x_test, t_test, batch_size)
    
    if test_acc>max_acc:
        max_acc = test_acc 
        net.save('./mnist_sample/')                        #Save the parameters
    
    print("│Epoch {:<4} Test Acc  :{:<5}│  ".format(j+1,str(test_acc*100)[:5]))
    print("│           Cost Time :{:<5}│  ".format(str(cost)[:5]))
    
print('└───────────────────────────┘  ')

net.update('./mnist_sample/weight')
print("Final Accuracy: %2f%%"%(100*net.accuracy(x_test, t_test)))