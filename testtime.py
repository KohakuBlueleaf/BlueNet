# coding: utf-8
from time import time
import numpy as np
from config import Network as model
from config import batch_size
from Network import Net

shape = (3,32,32)

start = time()
net = Net(model,data_shape=shape)
print('Initializing time:',time()-start)

data = np.random.randn(10,shape[0],shape[1],shape[2])
test = np.random.randn(10,10)

start = time()
net.train(data,test)
net.loss(data,test)
print('Sec per train step(Batch size 10):',(time()-start))
input()
