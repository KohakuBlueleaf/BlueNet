# coding: utf-8
import numpy as np

from mnist import load_mnist
from emnist import load_emnist
from cifar import load_cifar
from Network import Net
#import config of network
from config import iters_num,batch_size,optimizer,rate,init_std,AF
from config import Network as model

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True,flatten=False)#Load the database

net = Net(model,init_std,AF,rate,optimizer)
net.update() #load the parameters

print(net.accuracy(x_test,t_test)*100)
#print(net.loss(x_test,t_test))