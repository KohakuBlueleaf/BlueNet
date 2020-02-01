# coding: utf-8
import sys
sys.path.append("..")
from time import time
from setting import np
from config import VGG16 as model
from config import batch_size
from Network import Net

shape = (3,224,224)

start = time()
net = Net(model,data_shape=shape)
print('Initializing time:',time()-start)

data = np.random.randn(batch_size,shape[0],shape[1],shape[2]).astype(np.float32)
test = np.random.randn(batch_size,1000).astype(np.float32)

start = time()
for i in range(20):
	net.train(data,test)
print('Sec per train step:',((time()-start)/20))

#print(net.process(data).dtype)
input()
