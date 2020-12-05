import numpy as np
import cupy as cp
from matplotlib import pyplot as plt 
from time import time

## Import bluenet
from bluenet.network import Net
from bluenet.layer import *
from bluenet.activation import *
from bluenet.setting import _np

model =[
      Conv({'f_num':10, 'f_size':3, 'pad':1, 'stride':1}),
      Conv({'f_num':10, 'f_size':3, 'pad':1, 'stride':1}),
      Conv({'f_num':10, 'f_size':3, 'pad':1, 'stride':1}),
      Conv({'f_num':10, 'f_size':3, 'pad':1, 'stride':1}),
      Conv({'f_num':10, 'f_size':3, 'pad':1, 'stride':1}),
      Conv({'f_num':10, 'f_size':3, 'pad':1, 'stride':1}),
      Conv({'f_num':10, 'f_size':3, 'pad':1, 'stride':1}),
      Conv({'f_num':10, 'f_size':3, 'pad':1, 'stride':1}),
      Conv({'f_num':10, 'f_size':3, 'pad':1, 'stride':1}),
      Conv({'f_num':10, 'f_size':3, 'pad':1, 'stride':1}),
      Conv({'f_num':10, 'f_size':3, 'pad':1, 'stride':1}),
      Conv({'f_num':10, 'f_size':3, 'pad':1, 'stride':1}),
      Conv({'f_num':10, 'f_size':3, 'pad':1, 'stride':1}),
      Conv({'f_num':10, 'f_size':3, 'pad':1, 'stride':1}),
      Conv({'f_num':10, 'f_size':3, 'pad':1, 'stride':1}),
]

##Initialize the neural network
net = Net(model)
net.initialize(shape=(10,10,10), af=GELU, init_mode='orthogonal')
#Run the LSUV initialize
net.LSUVlize(4,3)

#Set the parameters for the test
r,c = 3,5
hist_range = 0.01
hist_min = -0.2
hist_max = hist_min+2.1
hist_bins = np.arange(hist_min,hist_max,hist_range)[:-1]
num = 10

#set the parameters for plot
fig, axes = plt.subplots(r,c)
mng = plt.get_current_fig_manager()

fig.tight_layout(pad=0.2)
plt.subplots_adjust(left=0.05, right=0.98, top=0.955, bottom=0.04, wspace=0.42, hspace=0.35)

#test
temp = net.test_gradient()
all_data = [_np.zeros((int((hist_max-hist_min)/hist_range)-1)) for i in range(len(temp))]

start = time()
for i in range(num):
  print(f'{i+1}/{num}',end='\r')
  temp = net.test_gradient(hist=True, hist_set=(hist_min, hist_max, hist_range), p=i==0)
  for i in range(len(all_data)):
    all_data[i] += temp[i]


i = 1
for data in all_data:
  print(f'{i}/{len(all_data)}      ',end='\r')
  plt.subplot(r,c,i)
  plt.bar(hist_bins, list(data), width=hist_range)
  plt.title(f'Layer {i}')
  i+=1


'''
all_data = [[] for i in range(len(temp))]

start = time()
num = 10
for i in range(num):
  print(f'{i+1}/{num}',end='\r')
  temp = net.test_gradient()
  all_data = [all_data[i]+list(temp[i].reshape(-1)) for i in range(len(temp))]

fig, axes = plt.subplots(r,c)
fig.tight_layout(pad=0.2)

i = 1
for data in all_data:
  print(f'{i}/{len(all_data)}      ',end='\r')
  plt.subplot(r,c,i)
  plt.hist(data, bins=np.arange(-1.05, 1.05, 0.01))
  plt.title(f'Layer {i}')
  i+=1
'''

#Show
mng.window.state("zoomed")
plt.show()