# coding: utf-8
from time import time
import numpy as np
from config import Network as model
from Network import Net

net = Net(model,data_shape=(1,28,28))
data = np.random.randn(100,1,28,28)
start = time()
A = net.process(data)
print(time()-start)
