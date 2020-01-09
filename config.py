# coding: utf-8
from optimizer import *
from layer import *
from Activation import *

#Hyperparameters
rate=0.001
batch_size=60
iters_num=100
init_std=0.01

#function for network
optimizer=Adam
AF = Elu

#The model of network
Network = 	[
			Flatten(),
			Dense(output_size=500),
			BatchNorm(),
			
			ResLayerV2([Dense(output_size=500),Dense(output_size=500)]),
			ResLayerV2([Dense(output_size=500),Dense(output_size=500)]),
			ResLayerV2([Dense(output_size=500),Dense(output_size=500)]),
			
			Dense(output_size=10),
			
			SoftmaxWithLoss()
			
			]

VGG16=[
			Conv({'f_num':64, 'f_size':3, 'pad':1, 'stride':1}),
			Conv({'f_num':64, 'f_size':3, 'pad':1, 'stride':1}),
			Pool(pool_h=2, pool_w=2, stride=2),
			
			Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1}),
			Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1}),
			Pool(pool_h=2, pool_w=2, stride=2),
			
			Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
			Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
			Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
			Pool(pool_h=2, pool_w=2, stride=2),
			
			Conv({'f_num':512, 'f_size':3, 'pad':1, 'stride':1}),
			Conv({'f_num':512, 'f_size':3, 'pad':1, 'stride':1}),
			Conv({'f_num':512, 'f_size':3, 'pad':1, 'stride':1}),
			Pool(pool_h=2, pool_w=2, stride=2),

			Conv({'f_num':512, 'f_size':3, 'pad':1, 'stride':1}),
			Conv({'f_num':512, 'f_size':3, 'pad':1, 'stride':1}),
			Conv({'f_num':512, 'f_size':3, 'pad':1, 'stride':1}),
			Pool(pool_h=2, pool_w=2, stride=2),
			
			Flatten(),
			
			Dense(output_size=4096),
			Dense(output_size=4096),
			Dense(output_size=1000),
			
			SoftmaxWithLoss()
			]
#Usual model
'''
VGG11=[
			Conv({'f_num':64, 'f_size':3, 'pad':1, 'stride':1}),
			Pool(pool_h=2, pool_w=2, stride=2),
			
			Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1}),
			Pool(pool_h=2, pool_w=2, stride=2),

			Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
			Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
			Pool(pool_h=2, pool_w=2, stride=2),

			Conv({'f_num':512, 'f_size':3, 'pad':1, 'stride':1}),
			Conv({'f_num':512, 'f_size':3, 'pad':1, 'stride':1}),
			Pool(pool_h=2, pool_w=2, stride=2),

			Conv({'f_num':512, 'f_size':3, 'pad':1, 'stride':1}),
			Conv({'f_num':512, 'f_size':3, 'pad':1, 'stride':1}),
			Pool(pool_h=2, pool_w=2, stride=2),
			
			Flatten(),
			
			Dense(output_size=4096),
			Dense(output_size=4096),
			Dense(output_size=1000),
			
			SoftmaxWithLoss()
			]

VGG13=[
			Conv({'f_num':64, 'f_size':3, 'pad':1, 'stride':1}),
			Conv({'f_num':64, 'f_size':3, 'pad':1, 'stride':1}),
			Pool(pool_h=2, pool_w=2, stride=2),
			
			Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1}),
			Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1}),
			Pool(pool_h=2, pool_w=2, stride=2),

			Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
			Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
			Pool(pool_h=2, pool_w=2, stride=2),

			Conv({'f_num':512, 'f_size':3, 'pad':1, 'stride':1}),
			Conv({'f_num':512, 'f_size':3, 'pad':1, 'stride':1}),
			Pool(pool_h=2, pool_w=2, stride=2),

			Conv({'f_num':512, 'f_size':3, 'pad':1, 'stride':1}),
			Conv({'f_num':512, 'f_size':3, 'pad':1, 'stride':1}),
			Pool(pool_h=2, pool_w=2, stride=2),
			
			Flatten(),
			
			Dense(output_size=4096),
			Dense(output_size=4096),
			Dense(output_size=1000),
			
			SoftmaxWithLoss()
			]

VGG16=[
			Conv({'f_num':64, 'f_size':3, 'pad':1, 'stride':1}),
			Conv({'f_num':64, 'f_size':3, 'pad':1, 'stride':1}),
			Pool(pool_h=2, pool_w=2, stride=2),
			
			Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1}),
			Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1}),
			Pool(pool_h=2, pool_w=2, stride=2),
			
			Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
			Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
			Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
			Pool(pool_h=2, pool_w=2, stride=2),
			
			Conv({'f_num':512, 'f_size':3, 'pad':1, 'stride':1}),
			Conv({'f_num':512, 'f_size':3, 'pad':1, 'stride':1}),
			Conv({'f_num':512, 'f_size':3, 'pad':1, 'stride':1}),
			Pool(pool_h=2, pool_w=2, stride=2),

			Conv({'f_num':512, 'f_size':3, 'pad':1, 'stride':1}),
			Conv({'f_num':512, 'f_size':3, 'pad':1, 'stride':1}),
			Conv({'f_num':512, 'f_size':3, 'pad':1, 'stride':1}),
			Pool(pool_h=2, pool_w=2, stride=2),
			
			Flatten(),
			
			Dense(output_size=4096),
			Dense(output_size=4096),
			Dense(output_size=1000),
			
			SoftmaxWithLoss()
			]
			
VGG19=[
			Conv({'f_num':64, 'f_size':3, 'pad':1, 'stride':1}),
			Conv({'f_num':64, 'f_size':3, 'pad':1, 'stride':1}),
			Pool(pool_h=2, pool_w=2, stride=2),
			
			Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1}),
			Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1}),
			Pool(pool_h=2, pool_w=2, stride=2),
			
			Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
			Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
			Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
			Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
			Pool(pool_h=2, pool_w=2, stride=2),
			
			Conv({'f_num':512, 'f_size':3, 'pad':1, 'stride':1}),
			Conv({'f_num':512, 'f_size':3, 'pad':1, 'stride':1}),
			Conv({'f_num':512, 'f_size':3, 'pad':1, 'stride':1}),
			Conv({'f_num':512, 'f_size':3, 'pad':1, 'stride':1}),
			Pool(pool_h=2, pool_w=2, stride=2),

			Conv({'f_num':512, 'f_size':3, 'pad':1, 'stride':1}),
			Conv({'f_num':512, 'f_size':3, 'pad':1, 'stride':1}),
			Conv({'f_num':512, 'f_size':3, 'pad':1, 'stride':1}),
			Conv({'f_num':512, 'f_size':3, 'pad':1, 'stride':1}),
			Pool(pool_h=2, pool_w=2, stride=2),
			
			Flatten(),
			
			Dense(output_size=4096),
			Dense(output_size=4096),
			Dense(output_size=1000),
			
			SoftmaxWithLoss()
			]
'''