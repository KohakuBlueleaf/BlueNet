# coding: utf-8

from layer import *
from functions import *
from optimizer import *

import numpy as np
import sys


class Net:
	
	def __init__(self,network,init_std=0.01,AF=Elu,learning_rate=0.001,optimizer=Adam,data_shape=(1,28,28)):
		self.network = []
		for i in network:
			self.network.append(i)
		
		self.layers = len(network)
		Ini = True
		
		for i in self.network:
			if i.name == 'ConvNet' or i.name == 'DeConvNet' or i.name == 'Dense':
				if i.params['W1'] is None:
					Ini = False
		
		if not Ini:
			init = np.random.randn(data_shape[0],data_shape[1],data_shape[2])
			j = 0
			for i in range(self.layers):
				if j == 0:
					if self.network[i].name == 'ConvNet' or self.network[i].name == 'DeConvNet' or self.network[i].name == 'ResLayer':
						init = init.reshape(1,init.shape[0],init.shape[1],init.shape[2])
					elif self.network[i].name == 'Dense':
						init = init.reshape(1,init.size)
				
				self.network[i].shapeIn = init.shape[1:]
				if self.network[i].name == 'ConvNet' or self.network[i].name == 'DeConvNet':
					self.network[i].params['W1'] = init_std * np.random.randn(self.network[i].f_num, init.shape[1], self.network[i].f_size, self.network[i].f_size)
					self.network[i].AF = AF()
					self.network[i].optimizer = optimizer(learning_rate)
					out = self.network[i].forward(init)
					if self.network[i].name == 'ConvNet':
						self.network[i].flops = ((init.shape[1]*self.network[i].f_size**2))*out.shape[2]*out.shape[3]*out.shape[1]
						self.network[i].size = self.network[i].params['W1'].size+self.network[i].params['b1'].size
					else:
						self.network[i].flops = ((init.shape[1]*self.network[i].f_size**2)-1)*out.shape[2]*out.shape[3]*out.shape[1]
						self.network[i].size = self.network[i].params['W1'].size
					init = out	
					
				elif self.network[i].name == 'Dense':
					self.network[i].params['W1'] = init_std * np.random.randn(init.size, self.network[i].output_size)
					self.network[i].size = self.network[i].params['W1'].size+self.network[i].params['b1'].size
					self.network[i].AF = AF()
					self.network[i].optimizer = optimizer(learning_rate)
					out = self.network[i].forward(init)
					self.network[i].flops = init.shape[1]*self.network[i].output_size
					self.network[i].size = self.network[i].params['W1'].size+self.network[i].params['b1'].size
				
				elif self.network[i].name == 'ResLayer':
					self.network[i].AF = AF
					init = self.network[i].initial(init,init_std,learning_rate,AF,optimizer)

				
				else:
					try:
						self.network[i].optimizer = optimizer(learning_rate)
					
					except:
						pass
				
				if self.network[i].name != 'Softmax' and self.network[i].name != 'ResLayer' and self.network[i].name != 'ConvNet' and self.network[i].name != 'DeConvNet':
					init = self.network[i].forward(init)
				
				self.network[i].shapeOut = init.shape[1:]
				j += 1
		else:
			pass
	
	def print_size(self):
		total = 0
		total_f = 0 
		print("┌───────────┬───────┬──────────┬──────────────┬─────────────┐")
		print("│   Layer   │ GFLOPs│  Params  │   Shape(In)  │  Shape(Out) │")
		for i in self.network:
			total += i.size
			total_f += i.flops
			print("├───────────┼───────┼──────────┼──────────────┼─────────────┤")
			print("│{:^11}│{:^7.3f}│{:>10}│{:>14}│{:>13}│".format(i.name,i.flops/1000000000,i.size,str(i.shapeIn).replace(' ',''),str(i.shapeOut).replace(' ','')))
		
		print("├───────────┼───────┼──────────┼──────────────┼─────────────┤")
		print("│   Total   │{:^7.2f}│{:>10}│              │             │".format(total_f/1000000000,total))
		print("└───────────┴───────┴──────────┴──────────────┴─────────────┘")	
		
	def process(self,input):
		for i in self.network:
			if i.name != 'DropOut':
				if i.name != 'Softmax':
					input = i.forward(input)
				else:
					input = i.forward_without_loss(input)

		return input

	def forward(self,input,t):
		for i in range(self.layers):
			if self.network[i].name != 'Softmax':
				input = self.network[i].forward(input)
			else:
				input = self.network[i].forward(input,t)
		
		return input
		
	def backward(self,error):	
		self.network.reverse()
		for i in range(self.layers):
			error = self.network[i].backward(error)
		self.network.reverse()		
	
	def train(self,input,answer):
		t = answer
		error = self.forward(input, t)
		self.backward(error)	
		for i in range(self.layers):
			if self.network[i].name == 'ResLayer':
				self.network[i].train()
			
			else:
				try:
					self.network[i].optimizer.update(self.network[i].params,self.network[i].grad)
				except AttributeError:
					pass
		
		return error

	def back_train(self,error):
		self.network.reverse()
		for i in range(self.layers):
			try:
				error = self.network[i].backward(error)
			
			except AttributeError:
				pass
			#print(error.shape)
		self.network.reverse()		
		
		for i in range(self.layers):
			try:
				self.network[i].optimizer.update(self.network[i].params,self.network[i].grad)
		
			except AttributeError:
				pass
		
	def accuracy(self, x, t):
		ac = 0
		for i in range(x.shape[0]//10):
			batch = np.arange(i*10,10+i*10)
			
			x_batch = x[batch]
			t_batch = t[batch]
			
			y = self.process(x_batch)
			y = np.argmax(y, axis=1)
			tt = np.argmax(t_batch, axis=1)
			
			ac += np.sum(y == tt)
		accuracy = ac / x.shape[0]
		
		return accuracy
	
	def loss(self, x, t): #caculate the loss of the net(CEE)
		y = self.process(x)
		loss = cross_entropy_error(y, t)
		
		return loss
	
	def loss_MSE(self, x, t): #caculate the loss of the net(MSE)
		y = self.process(x)
		
		return mean_squared_error(y, t)	
	
	def update(self):
		for i in range(self.layers):
			try:
				self.network[i].load(str(i+1))
		
			except AttributeError:#AF pooling flatten
				pass
			
			except FileNotFoundError:#file not found(Conv)
				pass
	
	def save(self):
		for i in range(self.layers):
			try:
				self.network[i].save(str(i+1))
			
			except AttributeError: #AF pooling Flatten
				pass

