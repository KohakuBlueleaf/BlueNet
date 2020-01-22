# coding: utf-8

#Original
from layer import *
from functions import *
from optimizer import *

#native(or usual)
from setting import np
import sys,os
import numpy


path = "./weight"
if not os.path.isdir(path):
	os.mkdir(path)
path = "./weight/new"
if not os.path.isdir(path):
	os.mkdir(path)

rn = np.random.randn

class Net:
	
	def __init__(self,network,init_std=0.01,init_mode = 'normal',AF=Elu,LR=0.001,optimizer=Adam,data_shape = (3,224,224)):
		self.net = []
		for i in network:
			self.net.append(i)
		self.layers = len(network)		#amount of layers
		Ini = True						#Initialized or not
		
		#if any layer's parameters haven't been setted, set Ini=False to run the initial process
		for i in self.net:
			if i.name == 'ConvNet' or i.name == 'DeConvNet' or i.name == 'Dense':
				if i.params['W1'] is None:
					Ini = False
		
		#initial process
		if not Ini:
			init = rn(data_shape[0],data_shape[1],data_shape[2])		#data for initial
			j = 0
			for i in range(self.layers):
				name = self.net[i].name
				
				if j == 0:
					#set the shape of data. Falt the data if first layer is dense
					if name == 'ConvNet' or name == 'DeConvNet' or name == 'ResLayer':
						init = init.reshape(1,init.shape[0],init.shape[1],init.shape[2])
					elif name == 'Dense':
						init = init.reshape(1,init.size)
				
				self.net[i].shapeIn = init.shape[1:]					#save the input shape
				
				if init_mode == 'xaiver':
					init_std = 1/(init.size**0.5)
				
				#Convolution
				if name == 'ConvNet' or name == 'DeConvNet':
					self.net[i].optimizer = optimizer(LR)				#set Optimizer(see optimizer.py)
					
					#Convolution
					if name == 'ConvNet':
						FN, C, S = self.net[i].f_num, init.shape[1], self.net[i].f_size
						
						self.net[i].params['W1'] = init_std * rn(FN, C, S, S)					#weight's shape is (F_num,input_channel,F_size,F_Size)
						self.net[i].params['b1'] *= init_std
						out = self.net[i].forward(init)											#data to set next layer
						
						N, out_C, out_H, out_W = out.shape
						self.net[i].flops = ((C*S**2))*out_H*out_W*out_C						#caculate the FLOPs
						self.net[i].size = FN*C*S*S + FN										#caculate the amount of parameters
					
					#Transpose Convolution
					else:
						FN, C, S = self.net[i].f_num, init.shape[1], self.net[i].f_size
						
						self.net[i].params['W1'] = init_std * rn(C, FN, S, S)					#weight's shape is (Input_channel,F_Num,F_size,F_Size)
						self.net[i].params['b1'] *= init_std
						out = self.net[i].forward(init)											#data to set next layer
						
						N, out_C, out_H, out_W = out.shape
						self.net[i].flops = ((C*S**2)-1)*out_H*out_W*out_C						#caculate the FLOPs
						self.net[i].size = self.net[i].params['W1'].size						#caculate the amount of parameters
				
					init = out
				
				#Fully connected layer
				elif name == 'Dense':
					out_size = self.net[i].output_size
					self.net[i].params['W1'] = init_std*rn(init.size, out_size)					#weight's shape is (input_size,output_size)
					self.net[i].params['b1'] *= init_std										
					self.net[i].optimizer = optimizer(LR)										#set Optimizer
					self.net[i].flops = init.shape[1]*out_size									#caculate the FLOPs
					self.net[i].size = init.size*out_size + out_size							#caculate the amount of parameters
				
				#ResLayer(Block of ResNet)
				elif name == 'ResLayer':
					self.net[i].AF = AF															#set Activation Function
					init = self.net[i].initial(init,init_std,init_mode,AF,optimizer,LR)			#see layer.py(In fact the function is same as here)
				
				elif name == 'BatchNorm':
					self.net[i].optimizer = optimizer(LR)
				
				else:
					pass
				
				#these layers don't need to caculate the data for next layer so we just skip it
				if name != 'Softmax' and name != 'ResLayer' and name != 'ConvNet' and name != 'DeConvNet':
					init = self.net[i].forward(init)
				
				#save the output shape
				self.net[i].shapeOut = init.shape[1:]
				j += 1
		else:
			pass
	
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
		input = np.asarray(input)
		for i in range(self.layers):
			if self.net[i].name != 'DropOut':
				if self.net[i].name != 'Softmax':
					if self.net[i].name == 'BatchNorm':
						input = self.net[i].forward(input,False)
					else:	
						input = self.net[i].forward(input)
				else:
					input = self.net[i].forward(input,loss = False)

		return input
		
	#forward process. DropOut is set ON. SoftmaxWithLoss return the loss
	def forward(self,input,t=None):
		input = np.asarray(input)
		t = np.asarray(t)
		for i in range(self.layers):
			if self.net[i].name != 'Softmax':
				input = self.net[i].forward(input)
			else:
				input = self.net[i].forward(input,t)
		
		return input
	
	#Backpropagation (will save the gradients)
	def backward(self,error):
		self.net.reverse()
		
		#backpropagation in order
		for i in range(self.layers):
			error = self.net[i].backward(error)
		self.net.reverse()
		
		#return final error(for GAN's Discriminator or others)
		return(error)
	
	#Backward + train(Change the parameters)
	def back_train(self,error):
		error = self.backward(error)	#backpropagation first
		
		for i in range(self.layers):
			#call the train function in the ResLayer
			if self.net[i].name == 'ResLayer':
				self.net[i].train()
			else:
				try:
					self.net[i].optimizer.update(self.net[i].params,self.net[i].grad)
				
				#if the layer doesn't have hte optimizer, skip it.
				except AttributeError:
					pass
		
		return error
	
	#Train consist of forward, backtrain, call the optimizer
	def train(self,input,t):
		input = np.asarray(input)
		t = np.asarray(t)

		self.net.reverse()
		#if final layer is SoftmaxWithLoss
		if self.net[0].name == 'Softmax':
			self.net.reverse()
			error = self.forward(input, t)		#forward(get the loss)
			self.back_train(error)
			loss = error

		#if final layer isn't SoftmaxWithLoss
		else:
			self.net.reverse()
			y = self.forward(input, t)			#froward(get the answer)
			loss = cross_entropy_error(y,t)		#caculate the loss
			
			batch_size = t.shape[0]
			if self.t.size == self.y.size:		#if the size of y and t is the same
				dx = (y - t) / batch_size
			
			else:								#if not
				dx = y.copy()
				dx[np.arange(batch_size), t] -= 1
				dx = dx / batch_size
			self.back_train(dx)
		
		#回傳loss
		return error
	
	#caculate the accuracy of the net
	def accuracy(self, x, t, batch_size = 100):
		ac = 0									#amount of correct answer
		for i in range(x.shape[0]//batch_size):			#process 10 datas in a time
			batch = numpy.arange(i*batch_size, batch_size+i*batch_size)	#choose the data in order
			
			x_batch = np.asarray(x[batch])
			t_batch = np.asarray(t[batch])
			
			y = self.process(x_batch)
			y = np.argmax(y, axis=1)			
			tt = np.argmax(t_batch, axis=1)
			ac += np.sum(y == tt)				#save the amount of correct answer
			
		accuracy = ac / x.shape[0]
		
		return accuracy
	
	#caculate the loss of the net(CEE)
	def loss(self, x, t): 
		t = np.asarray(t)
		y = self.process(x)
		loss = cross_entropy_error(y, t)
		
		return loss
	
	#caculate the loss of the net(MSE)
	def loss_MSE(self, x, t): 
		t = np.asarray(t)
		y = self.process(x)
		
		return mean_squared_error(y, t)	
	
	#Load the parameters
	def update(self):
		for i in range(self.layers):
			#call every layer's load function
			try:
				self.net[i].load(str(i+1))
		
			except AttributeError:	#AF pooling flatten
				pass
			
			except FileNotFoundError:
				pass
	
	#Save the parameters
	def save(self):
		for i in range(self.layers):
			#call every layer's save function
			try:
				self.net[i].save(str(i+1))
			
			except AttributeError: 	#AF pooling Flatten
				pass

