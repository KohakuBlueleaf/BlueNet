# coding: utf-8
from optimizer import *
from layer import *
from Activation import *

#Hyperparameters
rate = 0.001
batch_size = 200
init_std = 0.05
init_mode = 'xaiver'

#function for network
optimizer = Adam
AF = GELU

#The model of network
Network = [	
			Conv({'f_num':32, 'f_size':3, 'pad':1, 'stride':1}),
			BatchNorm(),
			Conv({'f_num':32, 'f_size':3, 'pad':1, 'stride':1}),
			BatchNorm(),
			Conv({'f_num':32, 'f_size':3, 'pad':0, 'stride':1}),
			BatchNorm(),
			
			Conv({'f_num':64, 'f_size':3, 'pad':1, 'stride':1}),
			BatchNorm(),
			Conv({'f_num':64, 'f_size':3, 'pad':1, 'stride':1}),
			BatchNorm(),
			Conv({'f_num':64, 'f_size':3, 'pad':0, 'stride':1}),
			Pool(2,2,2),
			BatchNorm(),
			
			Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1}),
			BatchNorm(),
			Conv({'f_num':128, 'f_size':3, 'pad':0, 'stride':1}),
			Pool(2,2,2),
			BatchNorm(),
			
			Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
			BatchNorm(),
			Conv({'f_num':256, 'f_size':3, 'pad':0, 'stride':1}),
			Pool(2,2,2),
			BatchNorm(),
			Flatten(),
			Dropout(0.4),
			
			Dense(output_size=512),
			BatchNorm(),
			Dropout(0.4),
			
			Dense(output_size=10),
			SoftmaxWithLoss(),
			]