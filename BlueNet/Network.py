# coding: utf-8

#Original
from BlueNet.Optimizer import *
from BlueNet.Activation import *
from BlueNet.Functions import *
from BlueNet.Layer import Conv,DeConv
from BlueNet.UsualModel import LeNet

#native(or usual)
from BlueNet.setting import _np
from copy import copy,deepcopy
from time import time
import sys,os
import numpy

rn = _np.random.randn
D3 = {'Conv','DeConv','ResLayer','Flatten'}
D2 = {'Dense'}
PRE = {'Conv','DeConv','ResLayer','Softmax'}
tp = type

class Net:
	
	def __init__(self, network=LeNet, data_shape=(1,28,28), AF=Relu, optimizer=Adam, rate=0.001\
					, init_std=0.005, init_mode='normal', type=_np.float32):
		normalization=''
		self.net = []
		for i in network:
			self.net.append(i)
		self.layers = len(network)								#amount of layers
		self.optimizer = tp(optimizer())
		self.learing_rate = rate
		
		#initial process
		init = rn(data_shape[0],data_shape[1],data_shape[2])	#data for initial
		j = 0
		for i in range(self.layers):
			name = self.net[i].name
			
			if j == 0:
				#set the shape of data. Falt the data if first layer is dense
				if name in D3:
					init = init.reshape(1,init.shape[0],init.shape[1],init.shape[2])
				elif name in D2:
					init = init.reshape(1,init.size)
			
			self.net[i].shapeIn = init.shape[1:]				#save the input shape
			
			if init_mode == 'xaiver':
				init_std = 1/(init.size**0.5)
			
			self.net[i].optimizer = optimizer(rate)				#set Optimizer
			if not self.net[i].AF:
				self.net[i].AF = AF()
			self.net[i].type = type
			
			#Convolution
			if name == 'Conv' or name == 'DeConv':
				self.net[i].params['b'] *= init_std
				FN, C, S = self.net[i].f_num, init.shape[1], self.net[i].f_size

				#Convolution
				if name == 'Conv':
					#weight's shape is (F_num, input_channel, F_size, F_Size)
					self.net[i].params['W'] = init_std * rn(FN, C, S, S)
					out = self.net[i].forward(init)						#data to set next layer
					
					N, out_C, out_H, out_W = out.shape
					self.net[i].flops = ((C*S**2))*out_H*out_W*out_C	#caculate the FLOPs
					self.net[i].size = FN*C*S*S + FN					#caculate the amount of parameters
				
				#Transpose Convolution
				else:
					#weight's shape is (Input_channel,F_Num,F_size,F_Size)
					self.net[i].params['W'] = init_std * rn(C, FN, S, S)
					out = self.net[i].forward(init)						#data to set next layer
					
					N, out_C, out_H, out_W = out.shape
					self.net[i].flops = ((C*S**2)-1)*out_H*out_W*out_C	#caculate the FLOPs
					self.net[i].size = self.net[i].params['W'].size		#caculate the amount of parameters
			
				init = out
					
			elif name == 'BatchNorm':
				self.net[i].size = 2
			
			#ResLayer(Block of ResNet)
			elif name == 'ResLayer':
				self.net[i].AF = AF																#set Activation Function
				init = self.net[i].initial(init,init_std,init_mode,AF,optimizer,rate,type)		#see layer.py(In fact the function is same as here)
			
			elif name == 'TimeDense':
				T = init.shape[1]
				D = init.shape[2]
				H = self.net[i].output_size
				self.net[i].params['W'] = rn(D, H)*init_std
				self.net[i].params['b'] = _np.ones(H)*init_std
				self.net[i].flops = T*D*H+H														#caculate the FLOPs
				self.net[i].size = D*(H+1)
				
			elif name == 'TimeLSTM':
				T = init.shape[1]
				D = init.shape[2]
				H = self.net[i].node
				self.net[i].params['Wx'] = rn(D, 4*H)*init_std
				self.net[i].params['Wh'] = rn(H, 4*H)*init_std
				self.net[i].params['b'] = _np.ones(4*H)*init_std
				self.net[i].flops = T*D*4*H+T*H*4*H												#caculate the FLOPs
				self.net[i].size = (D+H+1)*4*H
			
			elif name == 'TimeGRU':
				T = init.shape[1]
				D = init.shape[2]
				H = self.net[i].node
				self.net[i].params['Wx'] = rn(D, 3*H)*init_std
				self.net[i].params['Wh'] = rn(H, 3*H)*init_std
				self.net[i].params['b'] = _np.ones(3*H)*init_std
				self.net[i].flops = T*D*3*H+T*H*3*H												#caculate the FLOPs
				self.net[i].size = (D+H+1)*3*H
			
			#Fully connected layer
			elif name == 'Dense':
				out_size = self.net[i].output_size
				self.net[i].params['W'] = init_std*rn(init.size, out_size)						#weight's shape is (input_size,output_size)
				self.net[i].params['b'] *= init_std
				self.net[i].params['b'] = self.net[i].params['b']
				self.net[i].flops = init.shape[1]*out_size										#caculate the FLOPs
				self.net[i].size = init.size*out_size + out_size								#caculate the amount of parameters
				
			#ResLayer(Block of ResNet)
			elif name == 'ResLayer':
				self.net[i].AF = AF																#set Activation Function
				init = self.net[i].initial(init,init_std,init_mode,AF,optimizer,rate,type)		#see layer.py(In fact the function is same as here)
			
			elif name == 'TimeLSTM':
				T = init.shape[1]
				D = init.shape[2]
				H = self.net[i].node
				self.net[i].params['Wx'] = rn(D, 4*H)*init_std
				self.net[i].params['Wh'] = rn(H, 4*H)*init_std
				self.net[i].params['b'] = _np.ones(4*H)*init_std
				self.net[i].flops = T*D*4*H+T*H*4*H												#caculate the FLOPs
				self.net[i].size = (D+H+1)*4*H
			
			elif name == 'TimeGRU':
				T = init.shape[1]
				D = init.shape[2]
				H = self.net[i].node
				self.net[i].params['Wx'] = rn(D, 3*H)*init_std
				self.net[i].params['Wh'] = rn(H, 3*H)*init_std
				self.net[i].params['b'] = _np.ones(3*H)*init_std
				self.net[i].flops = T*D*3*H+T*H*3*H												#caculate the FLOPs
				self.net[i].size = (D+H+1)*3*H
			
			else:
				pass
			
			#these layers don't need to caculate the data for next layer so we just skip it
			if name not in PRE:
				try:
					init = self.net[i].forward(init)
				except:
					print(init.shape)
					print(self.net[i].params['W'].shape)
			
			#save the output shape
			self.net[i].shapeOut = init.shape[1:]
			j += 1
		
		for i in range(self.layers):
			try:
				for x in self.net[i].params.keys():
					try:
						self.net[i].params[x] = self.net[i].params[x].astype(type)
					except AttributeError:
						pass
			except:
				pass
	
	def __copy__(self):
		new = Net()
		new.net = deepcopy(self.net)
		new.layers = self.layers
		
		return new
	
	#print the model(About layer/amount of parameter/FLOPS...)
	def print_size(self):
		total = 0		#Total amount of parameters
		total_f = 0 	#Total FLOPs
		
		#print the table
		print("┌───────────┬───────┬──────────┬──────────────┬─────────────┐")
		print("│   Layer   │ GFLOPs│  Params  │   Shape(In)  │  Shape(Out) │")
		for i in self.net:
			try:
				total += i.size
				total_f += i.flops
				print("├───────────┼───────┼──────────┼──────────────┼─────────────┤")
				print("│{:^11}│{:^7.3f}│{:>10}│{:>14}│{:>13}│".format(i.name,i.flops/1000000000,i.size,str(i.shapeIn).replace(' ',''),str(i.shapeOut).replace(' ','')))
			except AttributeError:
				pass
				
		print("├───────────┼───────┼──────────┼──────────────┼─────────────┤")
		print("│   Total   │{:^7.2f}│{:>10}│              │             │".format(total_f/1000000000,total))
		print("└───────────┴───────┴──────────┴──────────────┴─────────────┘")	

	#forward process. DropOut is set OFF. SoftmaxWithLoss return the answer
	def process(self,input):
		input = _np.asarray(input)
		
		for i in range(self.layers):
			if self.net[i].name == 'DropOut':continue
			
			if self.net[i].name == 'Softmax':
				input = self.net[i].forward(input,0,False)
			else:
				input = self.net[i].forward(input)
		
		return input
		
	#forward process. DropOut is set ON. SoftmaxWithLoss return the loss
	def forward(self, input, t=None, loss_function=None):
		output = self.process(_np.asarray(input))
				
		if loss_function is not None and t is not None:
			t = _np.asarray(t)
			return input,loss_function(input,t)
		
		return output
	
	#Backpropagation (will save the gradients)
	def backward(self,error):
		self.net.reverse()
		
		#backpropagation in order
		for i in range(self.layers):
			error = self.net[i].backward(error)
		self.net.reverse()
		
		#return final error(for GAN's Discriminator or others)
		return error
	
	#Backward + train(Change the parameters)
	def back_train(self,error):
		error = self.backward(error)					#backpropagation first
		
		for i in range(self.layers):
			#call the train function
			self.net[i].train()
		
		return error
	
	#Train consist of forward, backtrain, call the optimizer
	def train(self, input, t, loss_function=cross_entropy_error):
		output = self.forward(input, t)			#forward
		t = _np.asarray(t)
		
		try:
			loss = loss_function(output,t)
		except:
			print(type(t), type(output))
		
		error = (output-t)/t.shape[0]
		self.back_train(error)
		
		return loss
	
	def pre_train_for_conv(self, all_data, batch_size=100):
		all_data = numpy.asarray(all_data[:10000])
		
		train_size = all_data.shape[0]
		iter_num = max(1, train_size//batch_size)
		if iter_num==1:
			batch_size = train_size
		
		layer_num=1
		print('\n┌────────────────────────────┐  ')
		print('│ Pre training start	     │  ')
		for i in range(len(self.net)):
			layer = self.net[i]
			layer.optimizer = self.optimizer()
			start = time()
			
			if layer.name=='Conv':
				print("│ ===========================│  ")
				
				other_layers = self.net[:i]
				init = _np.asarray(all_data[:1])
				for other_layer in other_layers:
					init = other_layer.forward(init)
				
				if layer.f_size == 2*layer.pad+1:
					pre_train_layer = Conv({'f_num':init.shape[1], 'f_size':layer.f_size, 'stride':layer.stride, 'pad':layer.pad})
				else:
					size = layer.f_size-2*layer.pad
					pre_train_layer = DeConv({'f_num':init.shape[1], 'f_size':size, 'stride':layer.stride})
				
				pre_train_layer.AF = type(layer.AF)()
				pre_train_layer.optimizer = type(layer.optimizer)()
				
				for _ in range(iter_num):
					data_mask = numpy.random.choice(train_size, batch_size)
					data = _np.asarray(all_data[data_mask])
					
					for other_layer in other_layers:
						data = other_layer.forward(data)
						
					#forward and backward
					forward = pre_train_layer.forward(layer.forward(data))
					error = (forward-data)/batch_size
					layer.backward(pre_train_layer.backward(error))
					
					#train
					layer.train()
					pre_train_layer.train()
					
					loss = RMS(data,forward)
					print('│ Layer {:<4}  Loss     :{:<5}│  '.format(i+1,str(loss)[:5]), end='\r')
				
				if i==0 or i==1:
					layer.optimizer = self.optimizer(0)
				else:
					layer.optimizer = self.optimizer(self.learing_rate)
			else:
				layer_num += 1
				continue
			
			end = time()
			cost = end-start
			
			print("│ Layer {:<4}       Loss:{:<5}│  ".format(i+1,str(loss)[:5]))
			print("│             Cost Time:{:<5}│  ".format(str(cost)[:5]))
		
		print('└────────────────────────────┘  ')

	def reset(self):
		for i in range(self.layers):
			if self.net[i].name == 'TimeLSTM':
				self.net[i].reset_state()
	
	#caculate the accuracy of the net
	def accuracy(self, x, t, batch_size = 100, print_the_result = False):
		ac = 0																#amount of correct answer
		for i in range(x.shape[0]//batch_size):								#process 10 datas in a time
			batch = numpy.arange(i*batch_size, batch_size+i*batch_size)		#choose the data in order
			
			x_batch = _np.asarray(x[batch])
			t_batch = _np.asarray(t[batch])
			
			y = self.process(x_batch)
			y = _np.argmax(y, axis=1)			
			tt = _np.argmax(t_batch, axis=1)
			ac += _np.sum(y == tt)											#save the amount of correct answer
			
		accuracy = ac / x.shape[0]
		if print_the_result:
			print(ac,'/',x.shape[0],sep='')
		
		return accuracy
	
	#caculate the loss of the net(CEE)
	def loss(self, x, t): 
		t = _np.asarray(t)
		y = self.process(x)
		loss = cross_entropy_error(y, t)
		
		return loss
	
	#caculate the loss of the net(MSE)
	def loss_MSE(self, x, t): 
		t = _np.asarray(t)
		y = self.process(x)
		
		return mean_squared_error(y, t)	
	
	def loss_RMS(self, x, t):
		t = _np.asarray(t)
		y = self.process(x)
		
		return RMS(y,t)
	
	#Load the parameters
	def update(self,folder='./'):
		for i in range(self.layers):
			#call every layer's load function
			try:
				self.net[i].load(str(i+1),folder)
			except FileNotFoundError:
				pass
	
	#Save the parameters
	def save(self,folder='./'):
		path = f"{folder}"
		if not os.path.isdir(path):
			os.mkdir(path)
		path = f"{folder}weight"
		if not os.path.isdir(path):
			os.mkdir(path)
		
		for i in range(self.layers):
			self.net[i].save(str(i+1),folder)
