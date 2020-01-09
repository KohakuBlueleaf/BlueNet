# coding: utf-8
import sys, os
sys.path.append(os.pardir) 
import numpy as np
from layer import *
from functions import *
from optimizer import *
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
			for i in range(len(self.network)):
				if i == 0:
					if self.network[i].name == 'ConvNet' or self.network[i].name == 'DeConvNet' or self.network[i].name == 'ResLayer':
						init = init.reshape(1,init.shape[0],init.shape[1],init.shape[2])
					elif self.network[i].name == 'Dense':
						init = init.reshape(1,init.size)
				
				self.network[i].shapeIn = init.shape[1:]
				if self.network[i].name == 'ConvNet' or self.network[i].name == 'DeConvNet':
					self.network[i].params['W1'] = init_std * np.random.randn(self.network[i].f_num, init.shape[1], self.network[i].f_size, self.network[i].f_size)
					self.network[i].AF = AF()
					self.network[i].optimizer = optimizer(learning_rate)
					try:
						self.network[i].size = self.network[i].params['W1'].size+self.network[i].params['b1'].size
					
					except:
						self.network[i].size = self.network[i].params['W1'].size
				
				elif self.network[i].name == 'Dense':
					self.network[i].params['W1'] = init_std * np.random.randn(init.size, self.network[i].output_size)
					self.network[i].size = self.network[i].params['W1'].size+self.network[i].params['b1'].size
					self.network[i].AF = AF()
					self.network[i].optimizer = optimizer(learning_rate)
				
				elif self.network[i].name == 'ResLayer':
					self.network[i].AF = AF
					init = self.network[i].initial(init,init_std,learning_rate,AF,optimizer)
					if self.network[i].shapeIn != init.shape[1:]:
						print('Error:ResLayer#{:<d} output size error'.format(i+1))
						sys.exit()
					
				else:
					try:
						i.optimizer = optimizer(learning_rate)
					
					except:
						pass
				
				if self.network[i].name != 'Softmax' and self.network[i].name != 'ResLayer':
					init = self.network[i].forward(init)
				
				self.network[i].shapeOut = init.shape[1:]
		else:
			pass
	
	def print_size(self):
		total = 0
		print("┌───────────┬───────────┬────────────────┬───────────────┐")
		print("│   Layer   │   Wsize   │    Shape(In)   │   Shape(Out)  │")
		for i in self.network:
			total += i.size
			print("├───────────┼───────────┼────────────────┼───────────────┤")
			print("│{:^11}│{:^11}│{:^16}│{:^15}│".format(i.name,i.size,str(i.shapeIn).replace(' ',''),str(i.shapeOut).replace(' ','')))
		
		print("├───────────┼───────────┼────────────────┼───────────────┤")
		print("│   Total   │{:^11}│                │               │".format(total))
		print("└───────────┴───────────┴────────────────┴───────────────┘")	
		
	def process(self,input):
		for i in range(self.layers):
			try:
				if self.network[i].name != 'DropOut':
					input = self.network[i].forward(input)
			
			except:
					input = self.network[i].forward_without_loss(input)

		return input

	def forward(self,input,t):
		for i in range(self.layers):
			try:
				input = self.network[i].forward(input)
			
			except:
				input = self.network[i].forward(input,t)
		
		return input

	def train(self,input,answer):
		t = answer
		for i in range(self.layers):
			try:
				input = self.network[i].forward(input)
		
			except:
					input = self.network[i].forward(input,t)
		error = input
		
		self.network.reverse()
		for i in range(self.layers):
			error = self.network[i].backward(error)
			#print(error.shape)
		
		self.network.reverse()		
		for i in range(self.layers):
			if self.network[i].name == 'ResLayer':
				self.network[i].train()
			
			else:
				try:
					self.network[i].optimizer.update(self.network[i].params,self.network[i].grad)
			
				except AttributeError:
					pass
	
	def backward(self,error):	
		self.network.reverse()
		for i in range(self.layers):
			try:
				error = self.network[i].backward(error)
			
			except AttributeError:
				pass
		self.network.reverse()		
		
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
				self.network[i].AdaGrad.update(self.network[i].params,self.network[i].grad)
		
			except AttributeError:
				pass
		
	def accuracy(self, x, t):
		ac = 0
		for i in range(x.shape[0]//100):
			batch = np.arange(i*100,100+i*100)
			
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
			
			except OSError:#file not found(affine)
				pass
	
	def save(self):
		for i in range(self.layers):
			try:
				self.network[i].save(str(i+1))
			
			except AttributeError: #AF pooling Flatten
				pass

