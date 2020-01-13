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
optimizer = Adam
AF = Elu

#The model of network
Network = 	[
			Conv({'f_num':15, 'f_size':5, 'pad':0, 'stride':1}),
			Conv({'f_num':30, 'f_size':6, 'pad':0, 'stride':2}),
			Pool(pool_h=2, pool_w=2, stride=2),
			Flatten(),
			Dense(output_size=420),
			Dense(output_size=230),
			Dense(output_size=10),
			
			SoftmaxWithLoss()
			]

#Usual model

'''
VGG
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

VGG21=[
			Conv({'f_num':64, 'f_size':3, 'pad':1, 'stride':1}),
			Conv({'f_num':64, 'f_size':3, 'pad':1, 'stride':1}),
			Conv({'f_num':64, 'f_size':3, 'pad':1, 'stride':1}),
			Pool(pool_h=2, pool_w=2, stride=2),
			
			Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1}),
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
ResNet
'''	

ResNet18 = 	[
			Conv({'f_num':64, 'f_size':7, 'pad':3, 'stride':2}),
			Pool(pool_h=3, pool_w=3, stride=2, pad=1),
			
			ResLayerV2([Conv({'f_num':64, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':64, 'f_size':3, 'pad':1, 'stride':1})]),
			ResLayerV2([Conv({'f_num':64, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':64, 'f_size':3, 'pad':1, 'stride':1})]),

			Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':2}),
			Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1}),
			ResLayerV2([Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1})]),
			
			Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':2}),
			Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1})]),
			
			Conv({'f_num':512, 'f_size':3, 'pad':1, 'stride':2}),
			Conv({'f_num':512, 'f_size':3, 'pad':1, 'stride':1}),
			ResLayerV2([Conv({'f_num':512, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':512, 'f_size':3, 'pad':1, 'stride':1})]),

			PoolAvg(pool_h=2, pool_w=2, stride=2),
			Flatten(),
			
			Dense(output_size=1000),
			
			SoftmaxWithLoss()
			]

ResNet34 = 	[
			Conv({'f_num':64, 'f_size':7, 'pad':3, 'stride':2}),
			Pool(pool_h=3, pool_w=3, stride=2, pad=1),
			
			ResLayerV2([Conv({'f_num':64, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':64, 'f_size':3, 'pad':1, 'stride':1})]),
			ResLayerV2([Conv({'f_num':64, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':64, 'f_size':3, 'pad':1, 'stride':1})]),
			ResLayerV2([Conv({'f_num':64, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':64, 'f_size':3, 'pad':1, 'stride':1})]),

			Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':2}),
			Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1}),
			ResLayerV2([Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1})]),
			ResLayerV2([Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1})]),
			ResLayerV2([Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1})]),
			
			Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':2}),
			Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1})]),
			
			Conv({'f_num':512, 'f_size':3, 'pad':1, 'stride':2}),
			Conv({'f_num':512, 'f_size':3, 'pad':1, 'stride':1}),
			ResLayerV2([Conv({'f_num':512, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':512, 'f_size':3, 'pad':1, 'stride':1})]),
			ResLayerV2([Conv({'f_num':512, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':512, 'f_size':3, 'pad':1, 'stride':1})]),

			PoolAvg(pool_h=2, pool_w=2, stride=2),
			Flatten(),
			
			Dense(output_size=1000),
			
			SoftmaxWithLoss()
			]

ResNet50 = 	[
			Conv({'f_num':64, 'f_size':7, 'pad':3, 'stride':2}),
			Pool(pool_h=3, pool_w=3, stride=2, pad=1),
			
			ResLayerV2([Conv({'f_num':64, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':64, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':64, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':64, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':64, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':64, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1})]),
			
			Conv({'f_num':128, 'f_size':1, 'pad':0, 'stride':2}),
			Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1}),
			Conv({'f_num':512, 'f_size':1, 'pad':0, 'stride':1}),
			ResLayerV2([Conv({'f_num':128, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':512, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':128, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':512, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':128, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':512, 'f_size':1, 'pad':0, 'stride':1})]),
			
			Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':2}),
			Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
			Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1}),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			
			Conv({'f_num':512, 'f_size':1, 'pad':0, 'stride':2}),
			Conv({'f_num':512, 'f_size':3, 'pad':1, 'stride':1}),
			Conv({'f_num':2048, 'f_size':1, 'pad':0, 'stride':1}),
			ResLayerV2([Conv({'f_num':512, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':512, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':2048, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':512, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':512, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':2048, 'f_size':1, 'pad':0, 'stride':1})]),
			PoolAvg(pool_h=2, pool_w=2, stride=2),
			Flatten(),
			
			Dense(output_size=1000),
			
			SoftmaxWithLoss()
			]

ResNet101 = 	[
			Conv({'f_num':64, 'f_size':7, 'pad':3, 'stride':2}),
			Pool(pool_h=3, pool_w=3, stride=2, pad=1),
			
			ResLayerV2([Conv({'f_num':64, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':64, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':64, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':64, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':64, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':64, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1})]),
			
			Conv({'f_num':128, 'f_size':1, 'pad':0, 'stride':2}),
			Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1}),
			Conv({'f_num':512, 'f_size':1, 'pad':0, 'stride':1}),
			ResLayerV2([Conv({'f_num':128, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':512, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':128, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':512, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':128, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':512, 'f_size':1, 'pad':0, 'stride':1})]),
			
			Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':2}),
			Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
			Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1}),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			
			Conv({'f_num':512, 'f_size':1, 'pad':0, 'stride':2}),
			Conv({'f_num':512, 'f_size':3, 'pad':1, 'stride':1}),
			Conv({'f_num':2048, 'f_size':1, 'pad':0, 'stride':1}),
			ResLayerV2([Conv({'f_num':512, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':512, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':2048, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':512, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':512, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':2048, 'f_size':1, 'pad':0, 'stride':1})]),
			PoolAvg(pool_h=2, pool_w=2, stride=2),
			Flatten(),
			
			Dense(output_size=1000),
			
			SoftmaxWithLoss()
			]
			
ResNet152 = 	[
			Conv({'f_num':64, 'f_size':7, 'pad':3, 'stride':2}),
			Pool(pool_h=3, pool_w=3, stride=2, pad=1),
			
			ResLayerV2([Conv({'f_num':64, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':64, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':64, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':64, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':64, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':64, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1})]),
			
			Conv({'f_num':128, 'f_size':1, 'pad':0, 'stride':2}),
			Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1}),
			Conv({'f_num':512, 'f_size':1, 'pad':0, 'stride':1}),
			ResLayerV2([Conv({'f_num':128, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':512, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':128, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':512, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':128, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':512, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':128, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':512, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':128, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':512, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':128, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':512, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':128, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':512, 'f_size':1, 'pad':0, 'stride':1})]),
			
			Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':2}),
			Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
			Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1}),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':1024, 'f_size':1, 'pad':0, 'stride':1})]),
			
			Conv({'f_num':512, 'f_size':1, 'pad':0, 'stride':2}),
			Conv({'f_num':512, 'f_size':3, 'pad':1, 'stride':1}),
			Conv({'f_num':2048, 'f_size':1, 'pad':0, 'stride':1}),
			ResLayerV2([Conv({'f_num':512, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':512, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':2048, 'f_size':1, 'pad':0, 'stride':1})]),
			ResLayerV2([Conv({'f_num':512, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':512, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':2048, 'f_size':1, 'pad':0, 'stride':1})]),
			PoolAvg(pool_h=2, pool_w=2, stride=2),
			Flatten(),
			
			Dense(output_size=1000),
			
			SoftmaxWithLoss()
			]