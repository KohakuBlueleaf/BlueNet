import numpy as np
from matplotlib import pyplot as plt 

## Import bluenet
from bluenet.network import Net
from bluenet.layer import *
from bluenet.activation import *

model = [
			Conv({'f_num':64, 'f_size':3, 'pad':1, 'stride':1},batchnorm=False),
			Conv({'f_num':64, 'f_size':3, 'pad':1, 'stride':1},batchnorm=False),
			Pool(2,2,2),

			Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1},batchnorm=False),
			Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1},batchnorm=False),
			Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1},batchnorm=False),
			Pool(2,2,2),

			Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1},batchnorm=False),
			Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1},batchnorm=False),
			Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1},batchnorm=False),
			Pool(2,2,2),
			
			Flatten(),
			Dropout(0.35),

			Dense(512),
			Dropout(0.35),

			SoftmaxWithLoss()
		]

##Initialize the neural network     
net = Net(model)
net.initialize(shape=(3,32,32), af=Tanh, init_std=0.001, init_mode='orthogonal')
temp = net.test_gradient()
all_data = [[] for i in range(len(temp))]

num = 1
for i in range(num):
	print(f'{i+1}/{num}',end='\r')
	temp = net.test_gradient()
	all_data = [all_data[i]+list(temp[i].reshape(-1)) for i in range(len(all_data))]

i = 1
r,c = 2, 6
fig, axes = plt.subplots(r,c)
fig.tight_layout(pad=0.2)

for data in all_data:
	print(f'{i}/{len(all_data)}      ',end='\r')
	plt.subplot(r,c,i)
	plt.hist(data, bins=np.arange(-1.05, 1.05, 0.01))
	plt.title(f'Layer {i}')
	i+=1

plt.subplots_adjust(left=0.04, right=0.985, top=0.955, bottom=0.04, wspace=0.4, hspace=0.2)
mng = plt.get_current_fig_manager()
mng.window.state("zoomed")
plt.show()