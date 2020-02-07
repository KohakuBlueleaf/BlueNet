# coding: utf-8

import sys
sys.path.append("./database") 

#import Nueral Network Module
from Network import Net

#import the loader of dataset
from mnist import load_mnist
from emnist import load_emnist
from cifar import load_cifar
from cifar100 import load_cifar100

#import config of network
from config import batch_size,optimizer,rate,init_std,init_mode,AF
from config import Network as model
from setting import np as cp

#import Usual module
import numpy as np
import time
import matplotlib.pyplot as plt


#initialize
(x_train,t_train),(x_test,t_test) = load_emnist(True, False, True, False, 0, np.float32)#Load the database
									#normalize,flatten,one_hot_label,smooth
train_size = x_train.shape[0]
train_loss_list = []
train_acc_list = []
test_acc_list = []
iter_per_epoch = max((train_size // batch_size), 1)
p = iter_per_epoch//50
max_acc = 0

#Set the network
net = Net(model, init_std, init_mode, AF, rate, optimizer, x_train[0].shape, np.float32)
net.update() #load the parameters

#start
start = time.time()
max_acc = net.accuracy(x_test, t_test, 100)
train_acc = net.accuracy(x_train, t_train, 100)
print("Start Accuracy:\nTest: %.2f%% Train: %.2f%%  Time: %.2fsec\n"%(max_acc*100,train_acc*100,time.time()-start))
test_acc_list.append(max_acc)
train_acc_list.append(train_acc)

#Train
round = int(input('Epoch:'))
print('\n┌──────────────────────────┐  ')
for j in range(round):
	if j != 0:
		print("│ =========================│  ")
	
	for i in range(iter_per_epoch):
		net.reset()
		batch_mask = np.random.choice(train_size, batch_size) #Random choose data
		
		x_batch = x_train[batch_mask]
		t_batch = t_train[batch_mask]
		
		loss = net.train(x_batch, t_batch) 	#Train&Caculate the loss of the net
		train_loss_list.append(loss)
		if i%p == 0:
			print('│ Epoch %2d  Loss:%5f  │  '%(j+1,loss),end='\r',flush=True)
	#print('Round %d Save         '%(j+1))
	
	test_acc = net.accuracy(x_test, t_test, 100)		#Caculate the accuracy of the net
	test_acc_list.append(test_acc)
	train_acc = net.accuracy(x_train, t_train, 100)		#Caculate the accuracy of the net
	train_acc_list.append(train_acc)
	
	if test_acc>max_acc:
		max_acc = test_acc 
		net.save() #Save the parameters
	
	print("│ Epoch %2d  Test Acc:%.3f│  "%(j+1,test_acc*100))
	print("│          Train Acc:%.3f│  "%(train_acc*100))

print('└──────────────────────────┘  ')

#finish
net.update()
print("Final Accuracy: %2f%%"%(100*net.accuracy(x_test, t_test)))

#print the acc
markers = {'train': 'o', 'test': 's'}
x = np.arange(1,len(test_acc_list)+1)
plt.plot(x, test_acc_list, marker='s', label='test acc', markevery=1)
plt.plot(x, train_acc_list, marker='o', label='train acc', markevery=1)
plt.xlabel("Rounds")
plt.ylabel("accuracy(%)")
plt.ylim(0,1)
plt.legend(loc='lower right')
plt.subplots_adjust(left=0.05, bottom=0.06, right=0.98, top=0.99, wspace=0.20, hspace=0.20)
plt.show()

#print the loss
x = np.arange(1,len(train_loss_list)+1)
plt.plot(x, train_loss_list, marker='o',label='loss', markevery=1)
plt.xlabel("iterations")
plt.ylabel("loss(CEE)")
plt.ylim(min(train_loss_list)*(0.98), max(train_loss_list)*(1.02))
plt.legend(loc='lower right')
plt.subplots_adjust(left=0.05, bottom=0.06, right=0.98, top=0.99, wspace=0.20, hspace=0.20)
plt.show()
