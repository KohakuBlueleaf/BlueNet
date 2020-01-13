# coding: utf-8

#import Usual module
import numpy as np
import time
import matplotlib.pyplot as plt

#import Nueral Network Module
from layer import *
from optimizer import *
from Network import Net

#import the loader of dataset
from mnist import load_mnist
from cifar import load_cifar
from emnist import load_emnist

#import config of network
from config import iters_num,batch_size,optimizer,rate,init_std,AF
from config import Network as model


#initialize
(x_train,t_train),(x_test,t_test) = load_mnist(True, False, True, False)#Load the database
									#normalize,flatten,one_hot_label,smooth
train_size = x_train.shape[0]

net = Net(model, init_std, AF, rate, optimizer, (1,28,28))
net.update() #load the parameters

train_loss_list = []
train_acc_list = []
test_acc_list = []
round_per_epoch = max((train_size / batch_size)/iters_num, 1)
max_acc = 0


#start
'''
start = time.time()
max_acc = 100*net.accuracy(x_test, t_test)
print("Start Accuracy %2f"%(max_acc))
print("Cost time:%2f"%(time.time()-start))
'''

#Train
round = int(input('Round:'))	
for j in range(round):
	for i in range(iters_num):
		batch_mask = np.random.choice(train_size, batch_size) #Random choose
		x_batch = x_train[batch_mask]
		t_batch = t_train[batch_mask]
		
		loss = net.train(x_batch, t_batch) #Train&Caculate the loss of the net
		train_loss_list.append(loss)		
		print('Loss:%6f'%loss)
	net.save()
	print('Round %d Save'%(j+1))
	'''
	test_acc = 100*net.accuracy(x_test, t_test)	#Caculate the accuracyof the net
	test_acc_list.append(test_acc)
	if test_acc>max_acc:
		max_acc = test_acc 
		net.save() #Save the parameters
	print("          ┌──────────┐")
	print("Round %3d:│ test acc │"%(j+1))
	print("          │ %f│"%(test_acc))
	print("          └──────────┘")
	'''


#finish
net.update()
#print("Final Loss: %6f"%(net.loss(x_test,t_test)))
print("Final Accuracy: %2f%%"%(100*net.accuracy(x_test, t_test)))


#print the loss/acc
'''
markers = {'train': 'o', 'test': 's'}
x = np.arange(1,len(test_acc_list)+1)
plt.plot(x, test_acc_list, marker='s', label='test acc', markevery=1)
plt.xlabel("Rounds")
plt.ylabel("accuracy(%)")
plt.ylim(min(test_acc_list)*(0.998), max(test_acc_list)*(1.002))
plt.legend(loc='lower right')
plt.subplots_adjust(left=0.05, bottom=0.06, right=0.98, top=0.99, wspace=0.20, hspace=0.20)
plt.show()
'''

FD =  (max(train_loss_list)-min(train_loss_list))/100
x = np.arange(1,len(train_loss_list)+1)
plt.plot(x, train_loss_list, marker='o',label='loss', markevery=1)
plt.xlabel("iterations")
plt.ylabel("loss(CEE)")
plt.ylim(min(train_loss_list)*(0.98), max(train_loss_list)*(1.02))
plt.legend(loc='lower right')
plt.subplots_adjust(left=0.05, bottom=0.06, right=0.98, top=0.99, wspace=0.20, hspace=0.20)
plt.show()