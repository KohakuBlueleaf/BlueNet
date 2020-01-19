# coding: utf-8
from optimizer import *
from layer import *
from Activation import *

#Hyperparameters
rate = 0.001
batch_size = 200
init_std = 0.01
init_mode = 'xaiver'

#function for network
optimizer = Adam
AF = GELU

#The model of network
Network = 	[
			Conv({'f_num':16, 'f_size':7, 'pad':3, 'stride':1}),
			Pool(2,2,2),
			BatchNorm(),
			
			Conv({'f_num':32, 'f_size':5, 'pad':2, 'stride':1}),
			Pool(2,2,2),
			BatchNorm(),
			
			Conv({'f_num':64, 'f_size':3, 'pad':1, 'stride':1}),
			Pool(2,2,2),
			BatchNorm(),
			
			Conv({'f_num':640, 'f_size':3, 'pad':0, 'stride':1}),
			Flatten(),
			BatchNorm(),
			
			Dense(output_size=400),
			BatchNorm(),
			
			Dense(output_size=26),
			
			SoftmaxWithLoss()
			]
#Usual model



'''
LeNet
'''

LeNet = [
			Conv({'f_num':6, 'f_size':5, 'pad':2, 'stride':1}),
			Pool(2,2,2),
			Conv({'f_num':16, 'f_size':5, 'pad':0, 'stride':1}),
			Pool(2,2,2),
			Conv({'f_num':120, 'f_size':5, 'pad':0, 'stride':1}),
			Flatten(),
			Dense(output_size=84),
			Dense(output_size=10),
				
			SoftmaxWithLoss()
		]


'''
AlexNet
'''

AlexNet=[
			Conv({'f_num':96, 'f_size':11, 'pad':3, 'stride':4}),
			Pool(3,3,2),
			Conv({'f_num':256, 'f_size':5, 'pad':2, 'stride':1}),
			Pool(3,3,2),
			Conv({'f_num':384, 'f_size':3, 'pad':1, 'stride':1}),
			Conv({'f_num':384, 'f_size':3, 'pad':1, 'stride':1}),
			Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1}),
			Pool(3,3,2),
			Flatten(),
			Dense(output_size=4096),
			Dense(output_size=4096),
			Dense(output_size=1000),
			
			SoftmaxWithLoss()
			]

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

			ResLayerV2([Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':2}),
						Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1})]),
			ResLayerV2([Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1})]),
			
			ResLayerV2([Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':2}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1})]),
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':1}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1})]),
			
			ResLayerV2([Conv({'f_num':512, 'f_size':3, 'pad':1, 'stride':2}),
						Conv({'f_num':512, 'f_size':3, 'pad':1, 'stride':1})]),
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

			ResLayerV2([Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':2}),
						Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1})]),
			ResLayerV2([Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1})]),
			ResLayerV2([Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1})]),
			ResLayerV2([Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':128, 'f_size':3, 'pad':1, 'stride':1})]),
			
			ResLayerV2([Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':2}),
						Conv({'f_num':256, 'f_size':3, 'pad':1, 'stride':1})]),
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
			
			ResLayerV2([Conv({'f_num':512, 'f_size':3, 'pad':1, 'stride':2}),
						Conv({'f_num':512, 'f_size':3, 'pad':1, 'stride':1})]),
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
			
			ResLayerV2([Conv({'f_num':128, 'f_size':1, 'pad':0, 'stride':2}),
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
			
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':2}),
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
			
			ResLayerV2([Conv({'f_num':512, 'f_size':1, 'pad':0, 'stride':2}),
						Conv({'f_num':512, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':2048, 'f_size':1, 'pad':0, 'stride':1})]),
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

ResNet101 = [
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
			
			ResLayerV2([Conv({'f_num':128, 'f_size':1, 'pad':0, 'stride':2}),
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
			
			ResLayerV2([Conv({'f_num':256, 'f_size':1, 'pad':0, 'stride':2}),
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
			
			ResLayerV2([Conv({'f_num':512, 'f_size':1, 'pad':0, 'stride':2}),
						Conv({'f_num':512, 'f_size':3, 'pad':1, 'stride':1}),
						Conv({'f_num':2048, 'f_size':1, 'pad':0, 'stride':1})]),
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
			
ResNet152 = [
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