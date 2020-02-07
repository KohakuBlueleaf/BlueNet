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
Network = 	[
			Conv({'f_num':18, 'f_size':5, 'pad':2, 'stride':1}),
			BatchNorm(),
			Conv({'f_num':18, 'f_size':5, 'pad':2, 'stride':1}),
			Pool(2,2,2),
			
			Conv({'f_num':48, 'f_size':5, 'pad':0, 'stride':1}),
			Pool(2,2,2),
			
			Conv({'f_num':360, 'f_size':5, 'pad':0, 'stride':1}),
			Flatten(),
			Dropout(0.25),
			
			Dense(output_size=252),
			BatchNorm(),
			Dropout(0.25),
			
			Dense(output_size=26),	
			SoftmaxWithLoss()
		]