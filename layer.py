# coding: utf-8

#Original module
from functions import *				#im2col and col2im function(written by 斎藤康毅)
from optimizer import *				#SGD/Adam/optimizer/Momentum/RMSprop
from Activation import *			#ReLU/Elu/GELU/ISRL/ISRLU/Sigmoid/Softplus/Softsign/Tanh/Arctan		

#Native Module
import pickle
from setting import np
from setting import exp
import cupy as cp
import sys


class Dense:
	
	'''
	Full Conected Layer
	'''
	
	def __init__(self, output_size, AF=Sigmoid,learning_rate=0.01,optimizer=Adam):
		self.name = 'Dense'
		#initialize
		self.output_size = output_size
		self.shapeIn = None									#shape of data(IN)
		self.shapeOut = None								#shape of data(OUT)
		
		#params
		self.params = {}
		self.params['W1'] = None							#Weight
		self.params['b1'] = np.ones(output_size)			#Bias
		self.grad = {}										#Gradient of weight and bias
		
		#other
		self.AF = AF()										#activation function
		self.optimizer = optimizer(lr = learning_rate)		#Optimizer
		self.size = None									#amount of params(Weight+bias)
		self.flops = None									#FLOPs of this layer
		
		#data for backward
		self.x = None
	
	def process(self, x):
		a1 = np.dot(x, self.params['W1']) + self.params['b1']
		z1 = self.AF.process(a1)
		
		return z1
	
	def forward(self, x):
		self.x = x
		a1 = np.dot(x, self.params['W1']) + self.params['b1']
		z1 = self.AF.forward(a1)							#Activation
		
		return z1
	
	def backward(self,error):
		
		dy = error	
		da1 = self.AF.backward(dy)							#Backpropagation for Activation Function
		dx = np.dot(da1, self.params['W1'].T)				#BP for input
		
		self.grad['W1'] = np.dot(self.x.T,da1)				#BP for weight
		self.x = None
		self.grad['b1'] = np.sum(da1, axis=0)				#BP for bias

		return dx
	
	def save(self, name="Dense_W"):							#Save the parameters
		params = {}
		for key, val in self.params.items():
			params[key] = val
		
		with open('./weight/new/Dense_W_'+name, 'wb') as f:
			pickle.dump(params, f)
	
	def load(self, name="Dense_W"):							#Load the parameters
		with open('./weight/new/Dense_W_'+name, 'rb') as f:
			params = pickle.load(f)
		
		for key, val in params.items():
			if val.shape == self.params[key].shape: 
				try:
					self.params[key] = np.asarray(val)
				except:
					self.params[key] = cp.asnumpy(val)
			else:
				print('weight shape error')

class Conv:

	'''
	Convolution Layer
	'''
	
	def __init__(self,conv_param,init_std=1,AF=Elu(),rate=0.01,optimizer=Adam):
		self.name = 'ConvNet'
		#Initialize
		self.shapeIn = None
		self.shapeOut = None
		self.f_num = conv_param['f_num']						#amount of filters
		self.f_size = conv_param['f_size']						#Size of filters
		self.f_pad = conv_param['pad']							#Padding size
		self.f_stride = conv_param['stride']					#Step of filters
		
		#params
		self.params = {}
		self.params['W1'] = None								#Set by intial process(see Network.py)
		self.params['b1'] = np.ones(self.f_num)					#Bias
		self.grad = {}											#Gradient of parameters
		self.stride = self.f_stride								#step(stride)
		self.pad = self.f_pad									#padding
		
		#other
		self.AF = AF											#set the function
		self.optimizer = optimizer(lr = rate)					#set the optimizer
		self.size = 0											#Amonut of parameters
		self.flops = 0											#Computation
		
		#data for backpropagation
		self.x_shape = None   									#shape of input
		self.col = None											#colume of input
	
	def process(self, x):
		FN, C, FH, FW = self.params['W1'].shape
		N, C, H, W = x.shape
		
		out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
		out_w = 1 + int((W + 2*self.pad - FW) / self.stride)
		
		col = im2col(x, FH, FW, self.stride, self.pad)					#Change the image to colume
		col_W = self.params['W1'].reshape(FN, -1).T						#Change the filters to colume
		out = np.dot(col, col_W) + self.params['b1']
		out = self.AF.process(out)
		out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)	#change colume to image

		return out
	
	def forward(self, x):
		FN, C, FH, FW = self.params['W1'].shape
		N, C, H, W = x.shape
		
		out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
		out_w = 1 + int((W + 2*self.pad - FW) / self.stride)
		
		col = im2col(x, FH, FW, self.stride, self.pad)					#Change the image to colume
		col_W = self.params['W1'].reshape(FN, -1).T						#Change the filters to colume
		out = np.dot(col, col_W) + self.params['b1']
		out = self.AF.forward(out)
		out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)	#change colume to image
		
		self.x_shape = x.shape
		self.col = col

		return out
	
	def backward(self, dout):
		FN, C, FH, FW = self.params['W1'].shape
		
		dout = dout.transpose(0,2,3,1).reshape(-1, FN)					#change gradient to colume
		dout = self.AF.backward(dout)
		
		self.grad['b1'] = np.sum(dout, axis=0)
		self.grad['W1'] = np.dot(self.col.T, dout)
		self.col = None
		self.grad['W1'] = self.grad['W1'].transpose(1, 0).reshape(FN, C, FH, FW)
		
		dcol = np.dot(dout, self.params['W1'].reshape(FN, -1))
		dx = col2im(dcol, self.x_shape, FH, FW, self.stride, self.pad)

		return dx
	
	def save(self, name="Conv_W"):
		params = {}
		for key, val in self.params.items():
			params[key] = val
	
		with open('./weight/new/Conv_W_'+name, 'wb') as f:
			pickle.dump(params, f)
	
	def load(self, name="Conv_W"):
		with open('./weight/new/Conv_W_'+name, 'rb') as f:
			params = pickle.load(f)
		
		for key, val in params.items():
			if val.shape == self.params[key].shape: 
				try:
					self.params[key] = np.asarray(val)
				except:
					self.params[key] = cp.asnumpy(val)
			else:
				print('weight shape error')

class DeConv:

	'''
	Transpose Convolution Layer
	The forward of DeConv is same to Convolution's backward and backward is same to conv's forward too.
	'''
	
	def __init__(self,conv_param,init_std=1,AF=Elu(),learning_rate=0.01,optimizer=Adam):
		self.name = 'DeConvNet'
		#Initialize
		self.shapeIn = None
		self.shapeOut = None
		self.f_num = conv_param['f_num']						#Amount of filters
		self.f_size = conv_param['f_size']						#Filter size
		self.f_stride = conv_param['stride']					#Step
		
		#params
		self.params = {}
		self.params['W1'] = None								#Set by intial process(see Network.py)
		self.size = 0
		self.flops = 0
		self.grad = {}
		self.stride = self.f_stride
		
		#Activation function & optimizer
		self.AF = AF
		self.optimizer = optimizer(lr = learning_rate)
		
		#data for backward
		self.x = None   
		self.col = None
		self.col_W = None

	def process(self, x):
		FN, C, FH, FW = self.params['W1'].shape
		N, B, H, W = x.shape
		
		out_h = FH + int((H - 1) * self.stride)
		out_w = FW + int((W - 1) * self.stride)
		
		col = x.transpose(0,2,3,1).reshape(-1,FN)
		col_W = self.params['W1'].reshape(FN, -1)
		out = np.dot(col, col_W)
		out = self.AF.process(out)
		out = col2im(out, (N, C , out_h, out_w), FH, FW, self.stride, 0)

		return out
	
	def forward(self, x):
		FN, C, FH, FW = self.params['W1'].shape
		N, B, H, W = x.shape
		
		out_h = FH + int((H - 1) * self.stride)
		out_w = FW + int((W - 1) * self.stride)
		
		col = x.transpose(0,2,3,1).reshape(-1,FN)
		col_W = self.params['W1'].reshape(FN, -1)
		out = np.dot(col, col_W)
		out = self.AF.forward(out)
		out = col2im(out, (N, C , out_h, out_w), FH, FW, self.stride, 0)
		
		self.x_shape = x.shape
		self.col = col
		self.col_W = col_W

		return out
	
	def backward(self, dout):
		FN, C, FH, FW = self.params['W1'].shape
		N, C, H, W = dout.shape
		
		dout = im2col(dout, FH, FW, self.stride, 0)
		dout = self.AF.backward(dout)
		
		self.grad['W1'] = np.dot(self.col.T, dout)
		self.grad['W1'] = self.grad['W1'].transpose(1, 0).reshape(FN, C, FH, FW)
		
		dcol = np.dot(dout, self.col_W.T)
		dx = dcol.reshape((self.x_shape[0],self.x_shape[2],self.x_shape[3],-1)).transpose(0,3,1,2)

		return dx
	
	def save(self, name="DeConv_W"):
		params = {}
		for key, val in self.params.items():
			params[key] = val
	
		with open('./weight/new/DeConv_W_'+name, 'wb') as f:
			pickle.dump(params, f)
	
	def load(self, name="DeConv_W"):
		with open('./weight/new/DeConv_W_'+name, 'rb') as f:
			params = pickle.load(f)
		
		for key, val in params.items():
			if val.shape == self.params[key].shape: 
				try:
					self.params[key] = np.asarray(val)
				except:
					self.params[key] = cp.asnumpy(val)


class Pool:
	
	'''
	Max-Pooling
	A convolution layer that the filter is choose the biggest value
	'''
	
	def __init__(self, pool_h, pool_w, stride=1, pad=0):
		self.name = 'Max-Pool'
		#Initialize
		self.shapeIn = None
		self.shapeOut = None
		
		#Setting about pooling
		self.pool_h = pool_h				#Height of the region to pool
		self.pool_w = pool_w				#width of the region to pool
		self.stride = stride
		self.pad = pad
		
		#other
		self.size = 0
		self.flops = 0
		
		#data for backward
		self.x = None
		self.arg_max = None
	
	def process(self, x):
		N, C, H, W = x.shape
		
		out_h = int(1 + (H+self.pad*2 - self.pool_h) / self.stride)
		out_w = int(1 + (W+self.pad*2 - self.pool_w) / self.stride)
		
		col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
		col = col.reshape(-1, self.pool_h*self.pool_w)
		arg_max = np.argmax(col, axis=1)								#Choose the highest value
		out = np.max(col, axis=1)
		out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)		#Colume reshape to image

		return out
	
	def forward(self, x):
		N, C, H, W = x.shape
		
		out_h = int(1 + (H+self.pad*2 - self.pool_h) / self.stride)
		out_w = int(1 + (W+self.pad*2 - self.pool_w) / self.stride)
		
		col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
		col = col.reshape(-1, self.pool_h*self.pool_w)
		arg_max = np.argmax(col, axis=1)								#Choose the highest value
		out = np.max(col, axis=1)
		out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)		#Colume reshape to image
		
		self.x = x
		self.arg_max = arg_max

		return out
	
	def backward(self, dout):
		dout = dout.transpose(0, 2, 3, 1)
		pool_size = self.pool_h * self.pool_w
		
		dmax = np.zeros((dout.size, pool_size))
		dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
		dmax = dmax.reshape(dout.shape + (pool_size,)) 
		
		dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
		dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
		
		return dx


class PoolAvg:
	
	'''
	Max-Pooling
	A convolution layer that the filter is caclulate the average.
	'''
	
	def __init__(self, pool_h, pool_w, stride=1, pad=0):
		self.name = 'Avg-Pool'
		#Initialize
		self.shapeIn = None
		self.shapeOut = None
		
		#Setting about pooling
		self.pool_h = pool_h
		self.pool_w = pool_w
		self.stride = stride
		self.pad = pad
		
		#other
		self.size = 0
		self.flops = 0
		
		#data for backward
		self.x_shape = None
		self.arg_max = None
	
	def process(self, x):
		N, C, H, W = x.shape
		
		out_h = int(1 + (H - self.pool_h) / self.stride)
		out_w = int(1 + (W - self.pool_w) / self.stride)
		
		col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
		col = col.reshape(-1, self.pool_h*self.pool_w)
		out = np.average(col, axis=1)									#caculate the average value
		out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
		
		return out
	
	def forward(self, x):
		N, C, H, W = x.shape
		
		out_h = int(1 + (H - self.pool_h) / self.stride)
		out_w = int(1 + (W - self.pool_w) / self.stride)
		
		col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
		col = col.reshape(-1, self.pool_h*self.pool_w)
		out = np.average(col, axis=1)									#caculate the average value
		out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
		
		self.x_shape = x.shape
		
		return out

	def backward(self, dout):
		pool_size = self.pool_h * self.pool_w
		N, C = dout.shape[0], dout.shape[1]
		
		dout = dout.transpose(0, 2, 3, 1)
		dout = dout.repeat(pool_size).reshape(N,C,-1,pool_size)/pool_size
		dx = col2im(dout, self.x_shape, self.pool_h, self.pool_w, self.stride, self.pad)
		
		return dx


class Flatten:
	
	'''
	3D(2D)data to 1D
	'''
	
	def __init__(self):
		self.name='Flatten'
		#Initialize
		self.shapeIn = None
		self.shapeOut = None
		self.in_size = None
		
		self.size = 0
		self.flops = 0
	
	def process(self,x):
		
		return x.reshape((x.shape[0],-1))
	
	def forward(self,x):
		self.in_size=x.shape
		
		return x.reshape((x.shape[0],-1))
	
	def backward(self, dout):
		
		return dout.reshape(self.in_size)
		
		
class BFlatten:
	
	'''
	1D data to 3D(2D)
	'''
	
	def __init__(self,size):
		self.name='BFlatten'
		#Initialize
		self.shapeIn = None
		self.shapeOut = None
		self.in_size=None
		self.out_size=size
		
		self.size = 0
		self.flops = 0
	
	def process(self,x):

		return x.reshape((x.shape[0],self.out_size[0],self.out_size[1],self.out_size[2]))
	
	def forward(self,x):
		self.in_size=x.shape

		return x.reshape((x.shape[0],self.out_size[0],self.out_size[1],self.out_size[2]))
	
	def backward(self, dout):
		
		return dout.reshape(self.in_size)


class BatchNorm:
	
	'''
	BatchNormalization
	'''
	
	def __init__(self, gamma=1.0, beta=0.0, momentum=0.9,
				running_mean=None, running_var=None,
				optimizer = Adam, learning_rate=0.001):
		self.name = 'BatchNorm'
		#initialize
		self.params = {}
		self.params['gamma']= gamma
		self.params['beta'] = beta
		self.momentum = momentum
		self.input_shape = None # Conv is 4d(N C H W), FCN is 2d(N D)
		
		#Traning data
		self.running_mean = running_mean
		self.running_var = running_var  
		
		# backward data
		self.batch_size = None
		self.xc = None
		self.std = None
		self.grad = {}
		self.grad['gamma'] = None
		self.grad['beta'] = None
		
		#Trainer for gamma&data
		self.optimizer = optimizer(lr = learning_rate)
		self.train_flg = False
		
		#other
		self.shapeIn = None
		self.shapeOut = None
		self.size = 2
		self.flops = 0
	
	def process(self, x, train_flg=False):
		self.input_shape = x.shape
		self.train_flg = train_flg
		if x.ndim != 2:
			N, C, H, W = x.shape
			x = x.reshape(N, -1)

		out = self.__forward(x, train_flg)
		
		return out.reshape(*self.input_shape)
	
	def forward(self, x, train_flg=True):
		self.input_shape = x.shape
		self.train_flg = train_flg
		if x.ndim != 2:
			N, C, H, W = x.shape
			x = x.reshape(N, -1)

		out = self.__forward(x, train_flg)
		
		return out.reshape(*self.input_shape)
			
	def __forward(self, x, train_flg):
		gamma, beta = self.params['gamma'],self.params['beta']
		if self.running_mean is None:
			N, D = x.shape
			self.running_mean = np.zeros(D)
			self.running_var = np.zeros(D)

		if train_flg:			#If you want to train the BatchNormalization layer, train_flg must be True
			mu = x.mean(axis=0)
			xc = x - mu
			var = np.mean(xc**2, axis=0)
			std = np.sqrt(var + 10e-7)
			xn = xc / std
			
			self.batch_size = x.shape[0]
			self.xc = xc
			self.xn = xn
			self.std = std
			self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
			self.running_var = self.momentum * self.running_var + (1-self.momentum) * var			
		
		else:
			xc = x - self.running_mean
			xn = xc / ((np.sqrt(self.running_var + 10e-7)))
			
		out = gamma * xn + beta 
		
		return out

	def backward(self, dout):
		if dout.ndim != 2:
			N, C, H, W = dout.shape
			dout = dout.reshape(N, -1)

		dx = self.__backward(dout)
		dx = dx.reshape(*self.input_shape)
		
		return dx

	def __backward(self, dout):
		gamma = self.params['gamma']
		
		dbeta = dout.sum(axis=0)
		dgamma = np.sum(self.xn * dout, axis=0)
		dxn = gamma * dout
		dxc = dxn / self.std
		dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
		dvar = 0.5 * dstd / self.std
		dxc += (2.0 / self.batch_size) * self.xc * dvar
		dmu = np.sum(dxc, axis=0)
		dx = dxc - dmu / self.batch_size
		
		if self.train_flg:
			self.grad['gamma'] = dgamma
			self.grad['beta'] = dbeta
		else:
			self.grad['gamma'] = 0
			self.grad['beta'] = 0
		
		return dx
	
	def save(self, name="Batch_Norm"):
		params = {}
		for key, val in self.params.items():
			params[key] = val
	
		with open('./weight/new/Batch_Norm_'+name, 'wb') as f:
			pickle.dump(params, f)
	
	def load(self, name="Batch_Norm"):
		with open('./weight/new/Batch_Norm_'+name, 'rb') as f:
			params = pickle.load(f)
		
		for key, val in params.items():
			self.params[key] = val


class Dropout:
	
	def __init__(self, dropout_ratio=0.5):
		self.name='DropOut'
		self.dropout_ratio = dropout_ratio
		self.mask = None
		self.shapeIn = None
		self.shapeOut = None
		self.size = 0
		self.flops = 0
	
	def forward(self, x, train_flg=True):
		if train_flg:
			self.mask = np.random.rand(*x.shape) > self.dropout_ratio
			
			return x * self.mask
		else:
			
			return x * (1.0 - self.dropout_ratio)

	def backward(self, dout):
		
		return dout * self.mask


class ResLayer:
	
	def __init__(self,layer):
		self.name = 'ResLayer'
		#Initialize
		self.layer = layer
		self.AF = None
		self.layers = []
		
		#additional layer to fit the shape
		self.Conv = None
		self.use_conv = False
		self.pool = None
		self.use_pool = False
		
		#other
		self.size = 0
		self.flops = 0
		self.shapeOut = None
		self.shapeIn = None
		
	def initial(self,data,init_std,init_mode='normal',AF=Elu,optimizer=Adam,rate=0.001):
		init = data
		
		for i in range(len(self.layer)):
			if init_mode == 'xaiver':
				init_std = 1/(init.size**0.5)
			
			if self.layer[i].name == 'ConvNet':
				FN, C, S = self.layer[i].f_num, init.shape[1], self.layer[i].f_size
				
				#set the params
				self.layer[i].params['W1'] = init_std * np.random.randn(FN, C, S, S)
				self.layer[i].params['b1'] *= init_std
				out = self.layer[i].forward(init)
				
				#Caculate the FLOPs & Amount of params
				N, out_C, out_H, out_W = out.shape
				self.layer[i].flops = (C *S**2)*out_H*out_W*out_C
				self.layer[i].size = FN*C*S*S + FN
				self.flops += self.layer[i].flops
				self.size += self.layer[i].size
				
				init = out
				
			elif self.layer[i].name == 'Dense':
				if init.ndim!=2:
					init = init.reshape(init.shape[0],-1)
				
				#set the params
				out_size =  self.layer[i].output_size
				self.layer[i].params['W1'] = init_std * np.random.randn(init.size, out_size)
				self.layer[i].params['b1'] *= init_std
				
				#Caculate the FLOPs & Amount of params
				self.layer[i].size = init.size*out_size + out_size
				self.layer[i].flops = (init.shape[1])*out_size
				self.size += self.layer[i].size
				self.flops += self.layer[i].flops
				
				init = self.layer[i].forward(init)
		
		if init.ndim == 4:
			if init.shape[1] != data.shape[1]:
				FN, C = init.shape[1], data.shape[1]
				out_C, out_H, out_W = init.shape[1],data.shape[2],data.shape[3]
				
				#set the params
				self.Conv = Conv({'f_num':init.shape[1],'f_size':1,'pad':0,'stride':1})
				self.Conv.params['W1'] = init_std * np.random.randn(FN,C,1,1)
				self.Conv.params['b1'] *= init_std
				
				#set Activation Functions & optimizer
				self.Conv.AF = AF()
				self.Conv.optimizer = optimizer(rate)
				
				#Caculate the FLOPs & Amount of params
				self.Conv.size = FN*C+FN
				self.Conv.flops = (C *S**2)*out_H*out_W*out_C
				self.size += self.Conv.size
				self.flops += self.Conv.flops
				
				self.use_conv = True
					
			if init.shape[2] != data.shape[2]:
				if init.shape[2] == data.shape[2]//2:
					self.pool = Pool(2,2,2)
					self.use_pool = True
				
				elif init.shape[2] == (data.shape[2]//2)+1:
					self.pool = Pool(2,2,2,1)
					self.use_pool = True
				
				else:
					print(init.shape)
					print('Shape Error')
					sys.exit()
	
		for i in self.layer:
			i.AF = ID()
			i.optimizer = optimizer(rate)
			self.layers.append(i)
			self.layers.append(BatchNorm())
			self.layers.append(AF())
		
		self.layers.pop()
		
		return init
	
	def process(self,x):
		out = x
		for i in self.layers:
			out = i.process(out)
		
		if self.use_conv:
			if self.use_pool:
				temp = self.Conv.process(x)
				out2 = self.pool.process(temp)
			else:
				out2 = self.Conv.process(x)
		
		else:
			if self.use_pool:
				out2 = self.pool.process(x)
			else:
				pass
		
		return out+out2
	
	def forward(self,x):
		out = x
		for i in self.layers:
			out = i.forward(out)
		
		if self.use_conv:
			if self.use_pool:
				temp = self.Conv.forward(x)
				out2 = self.pool.forward(temp)
			else:
				out2 = self.Conv.forward(x)
		
		else:
			if self.use_pool:
				out2 = self.pool.forward(x)
			else:
				pass
		
		return out+out2
	
	def backward(self,dout):
		dx = dout
		
		self.layers.reverse()
		for i in self.layers:
			dx = i.backward(dx)
		self.layers.reverse()
		
		if self.use_conv:
			if self.use_pool:
				temp = self.pool.backward(dout)
				dx2 = self.Conv.backward(temp)
			else:
				dx2 = self.Conv.backward(dout)
		
		else:
			if self.use_pool:
				dx2 = self.pool.backward(dout)
			else:
				pass
			
		return dx+dx2
	
	def train(self):
		for i in self.layers:
			try:
				i.optimizer.update(i.params,i.grad)
			except AttributeError:
				pass
		
		if self.use_conv:
			self.Conv.optimizer.update(self.Conv.params,self.Conv.grad)
	
	def save(self, name="Res"):
		j = 1
		for i in self.layers:
			try:
				i.save(str(j)+'_Res_'+name)
			except AttributeError: #AF pooling Flatten
				pass
			
			j+=1
		
		if use_conv:
			self.Conv.save(str(j+1)+'_Res_'+name)
		
	def load(self, name="Res"):
		j = 1
		for i in self.layers:
			try:
				i.load(str(j)+'_Res_'+name)
			except AttributeError:#AF pooling flatten
				pass
			except FileNotFoundError:#file not found(Conv,Deconv,BN,Dense)
				pass
			
			j+=1	
	
		if self.use_conv:
			self.Conv.load(str(j+1)+'_Res_'+name)
		

class ResLayerV2:
	
	def __init__(self,layer):
		self.name = 'ResLayer'
		#initialize
		self.layer = layer
		self.AF = None
		self.layers = []
		
		#additional layer to fit the shape
		self.Conv = None
		self.use_conv = False
		self.pool = None
		self.use_pool = False
		
		#other
		self.size = 0
		self.flops = 0
		self.shapeOut = None
		self.shapeIn = None
		
	def initial(self,data,init_std,init_mode='normal',AF=Elu,optimizer=Adam,rate=0.001):
		init = data
		
		for i in range(len(self.layer)):
			if init_mode == 'xaiver':
				init_std = 1/(init.size**0.5)
			
			if self.layer[i].name == 'ConvNet':
				FN, C, S = self.layer[i].f_num, init.shape[1], self.layer[i].f_size
				
				#set the params
				self.layer[i].params['W1'] = init_std * np.random.randn(FN, C, S, S)
				self.layer[i].params['b1'] *= init_std
				out = self.layer[i].forward(init)
				
				#Caculate the FLOPs & Amount of params
				N, out_C, out_H, out_W = out.shape
				self.layer[i].flops = (C *S**2)*out_H*out_W*out_C
				self.layer[i].size = FN*C*S*S + FN
				self.flops += self.layer[i].flops
				self.size += self.layer[i].size
				
				init = out
				
			elif self.layer[i].name == 'Dense':
				if init.ndim!=2:
					init = init.reshape(init.shape[0],-1)
				
				#set the params
				out_size =  self.layer[i].output_size
				self.layer[i].params['W1'] = init_std * np.random.randn(init.size, out_size)
				self.layer[i].params['b1'] *= init_std
				
				#Caculate the FLOPs & Amount of params
				self.layer[i].size = init.size*out_size + out_size
				self.layer[i].flops = (init.shape[1])*out_size
				self.size += self.layer[i].size
				self.flops += self.layer[i].flops
				
				init = self.layer[i].forward(init)
		
		if init.ndim == 4:
			if init.shape[1] != data.shape[1]:
				FN, C = init.shape[1], data.shape[1]
				out_C, out_H, out_W = init.shape[1],data.shape[2],data.shape[3]
				
				#set the params
				self.Conv = Conv({'f_num':init.shape[1],'f_size':1,'pad':0,'stride':1})
				self.Conv.params['W1'] = init_std * np.random.randn(FN,C,1,1)
				self.Conv.params['b1'] *= init_std
				
				#set Activation Functions & optimizer
				self.Conv.AF = AF()
				self.Conv.optimizer = optimizer(rate)
				
				#Caculate the FLOPs & Amount of params
				self.Conv.size = FN*C+FN
				self.Conv.flops = (C *S**2)*out_H*out_W*out_C
				self.size += self.Conv.size
				self.flops += self.Conv.flops
				
				self.use_conv = True
					
			if init.shape[2] != data.shape[2]:
				if init.shape[2] == data.shape[2]//2:
					self.pool = Pool(2,2,2)
					self.use_pool = True
				
				elif init.shape[2] == (data.shape[2]//2)+1:
					self.pool = Pool(2,2,2,1)
					self.use_pool = True
				
				else:
					print(init.shape)
					print('Shape Error')
					sys.exit()
		
		for i in self.layer:
			i.AF = ID()
			i.optimizer = optimizer(rate)
			self.layers.append(BatchNorm(optimizer = optimizer, learning_rate = rate))
			self.layers.append(AF())
			self.layers.append(i)	

		return init
	
	def process(self,x):
		out = x
		for i in self.layers:
			out = i.process(out)
		
		if self.use_conv:
			if self.use_pool:
				temp = self.Conv.process(x)
				out2 = self.pool.process(temp)
			else:
				out2 = self.Conv.process(x)
		
		else:
			if self.use_pool:
				out2 = self.pool.process(x)
			else:
				pass
		
		return out+out2
	
	def forward(self,x):
		out = x
		for i in self.layers:
			out = i.forward(out)
		
		if self.use_conv:
			if self.use_pool:
				temp = self.Conv.forward(x)
				out2 = self.pool.forward(temp)
			else:
				out2 = self.Conv.forward(x)
		
		else:
			if self.use_pool:
				out2 = self.pool.forward(x)
			else:
				out2 = x
		
		return out+out2
	
	def backward(self,dout):
		dx = dout
		
		self.layers.reverse()
		for i in self.layers:
			dx = i.backward(dx)
		self.layers.reverse()
		
		if self.use_conv:
			if self.use_pool:
				temp = self.pool.backward(dout)
				dx2 = self.Conv.backward(temp)
			else:
				dx2 = self.Conv.backward(dout)
		
		else:
			if self.use_pool:
				dx2 = self.pool.backward(dout)
			else:
				dx2 = dout
			
		return dx+dx2
	
	def train(self):
		for i in self.layers:
			try:
				i.optimizer.update(i.params,i.grad)
			except AttributeError:
				pass
		
		if self.use_conv:
			self.Conv.optimizer.update(self.Conv.params,self.Conv.grad)
			
	def save(self, name="Res"):
		j = 1
		for i in self.layers:
			try:
				i.save(str(j)+'_Res_'+name)
			except AttributeError: 	#AF pooling Flatten
				pass
			
			j+=1
		
		if self.use_conv:
			self.Conv.save(str(j+1)+'_Res_'+name)
		
	def load(self, name="Res"):
		j = 1
		for i in self.layers:
			try:
				i.load(str(j)+'_Res_'+name)
			except AttributeError:		#AF pooling flatten
				pass
			except FileNotFoundError:	#file not found(Conv,Deconv,BN,Dense)
				pass
			
			j+=1
		
		if self.use_conv:
			self.Conv.load(str(j+1)+'_Res_'+name)



'''
Layer for RNN
'''


class Embedding:
	
	'''
	Word Embedding
	'''
	
	def __init__(self, W):
		self.params = [W]
		self.grads = [np.zeros_like(W)]
		self.idx = None
	
	def forward(self, idx):
		W, = self.params
		self.idx = idx
		out = W[idx]
		
		return out
	
	def backward(self):
		dW, = self.grads
		dW[...] = 0
		np.add.at(dW, self.idx, dout)
		
		return None


class TimeDense:
	
	def __init__(self,stateful=False,AF=Elu,optimizer=Adam,rate=0.001):
		self.name = 'TimeDense'
		#initialize
		self.dh = None
		
		#parameters
		self.params={}
		self.params['W1'] = None
		self.params['b1'] = None
		self.grad = {}
		self.grad['W1'] = None
		self.grad['b1'] = None	
		
		#other
		self.optimizer = optimizer(rate)
		self.AF = AF
		self.flops = 0
		self.size = 0
	
	def forward(self, xs):
		Wx, b = self.params['W1'],self.params['b1']
		N, T, D = xs.shape
		H = Wx.shape[1]
		
		self.layers = []
		hs = np.empty((N, T, H), dtype='f')
		
		for t in range(T):
			layer = Dense(H,AF=self.AF)
			layer.params['W1'] = Wx
			layer.params['b1'] = b
			self.h = layer.forward(xs[:, t, :])
			hs[:, t, :] = self.h
			
			self.layers.append(layer)
			
		return hs
		
	def backward(self,dhs):
		Wx, b = self.params['W1'],self.params['b1']
		N, T, H = dhs.shape
		D = Wx.shape[0]
		
		dxs = np.empty((N, T, D), dtype='f')
		self.grad['W1'] = np.zeros_like(Wx)
		self.grad['b1'] = np.zeros_like(b)
		
		for t in reversed(range(T)):
			layer = self.layers[t]
			dx = layer.backward(dhs[:, t, :])
			try:
				dxs[:, t, :] = dx
			except:
				print(dxs.shape)
			for i in layer.grad:
				self.grad[i] += layer.grad[i]
		
		return dxs

	def save(self, name):
		params = {}
		for key, val in self.params.items():
			params[key] = val
		with open('./weight/new/TimeDense_'+name, 'wb') as f:
			pickle.dump(params, f)
	
	def load(self, name):
		with open('./weight/new/TimeDense_'+name, 'rb') as f:
			params = pickle.load(f)
		
		for key, val in params.items():
			try:
				self.params[key] = np.asarray(val)
			except:
				self.params[key] = cp.asnumpy(val)
	
	
class LSTM:
	
	'''
	Long short-term memory
	'''
	
	def __init__(self,Wx,Wh,b):
		self.name = 'LSTM'
		#parameters and grads
		self.params = [Wx, Wh, b]
		self.grad = [np.zeros_like(i) for i in self.params]
		
		#gate
		self.sigm1 = Sigmoid()
		self.sigm2 = Sigmoid()
		self.sigm3 = Sigmoid()
		self.tanIn = Tanh()
		self.tanOu = Tanh()
		
		#data for backward
		self.In = None
		self.Hin = None
		self.Tout = None
		self.Tin = None
		self.Cin = None
		
		#other
		self.size = 0
		self.flops = 0
		
	def forward(self,input,Hin,C):
		Wx,Wh,b = self.params
		N, H = Hin.shape
		A = np.dot(input,Wx)+np.dot(Hin,Wh)+b
		
		#slice
		f = A[:, :H]
		g = A[:, H:2*H]
		i = A[:, 2*H:3*H]
		o = A[:, 3*H:]
		
		Sf = self.sigm1.forward(f)
		Tin = self.tanIn.forward(g)
		Si = self.sigm2.forward(i)
		So = self.sigm3.forward(o)
		Cout = C*Sf+Si*Tin
		Tout = self.tanOu.forward(Cout)
		Hout = Tout*So
		
		self.Hin = Hin
		self.In = input
		self.Cin = C
		self.Tout = Tout
		self.Tin = Tin

		return Hout,Cout
	
	def backward(self,dOut,dHout,dC):
		Wx,Wh,b = self.params
		
		dHout = dOut+dHout
		dTout = self.tanOu.backward(dHout*self.sigm3.out)
		dCin = (dC+dTout)*self.sigm1.out
		
		df = self.sigm1.backward((dC+dTout)*self.Cin)			#df
		dg = self.tanIn.backward((dC+dTout)*self.sigm2.out)		#dg
		di = self.sigm2.backward((dC+dTout)*self.Tin)			#di
		do = self.sigm3.backward(dOut*self.Tout)				#do
		
		dA = np.hstack((df,dg,di,do))
		self.grad[2] = np.hstack((np.sum(df,axis=0),np.sum(dg,axis=0),np.sum(di,axis=0),np.sum(do,axis=0)))
		self.grad[1] = np.dot(self.Hin.T, dA)
		self.grad[0] = np.dot(self.In.T, dA)

		dHin = np.dot(dA, Wh.T)
		dIn = np.dot(dA, Wx.T)
	
		return dIn,dHin,dCin


class TimeLSTM:
	
	def __init__(self,stateful=False,optimizer=Adam,rate=0.001):
		self.name = 'TimeLSTM'
		#initialize
		self.h, self.c = None,None
		self.dh = None
		self.stateful = stateful
		
		#parameters
		self.params={}
		self.params['Wx'] = None
		self.params['Wh'] = None
		self.params['b'] = None
		self.grad = {}
		self.grad['Wx'] = None
		self.grad['Wh'] = None
		self.grad['b'] = None	
		
		self.flops = 0
		self.size = 0
		self.optimizer = optimizer(rate)
		
	def forward(self, xs):
		Wx, Wh, b = self.params['Wx'],self.params['Wh'],self.params['b']
		N, T, D = xs.shape
		H = Wh.shape[0]
		
		self.layers = []
		hs = np.empty((N, T, H), dtype='f')
		
		if not self.stateful or self.h is None:
			self.h = np.zeros((N,H), dtype='f')
		if not self.stateful or self.c is None:
			self.c = np.zeros((N,H), dtype='f')

		for t in range(T):
			layer = LSTM(Wx,Wh,b)
			self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c)
			hs[:, t, :] = self.h
			
			self.layers.append(layer)
			
		return hs
		
	def backward(self,dhs):
		Wx, Wh, b = self.params['Wx'],self.params['Wh'],self.params['b']
		N, T, H = dhs.shape
		D = Wx.shape[0]
		
		dxs = np.empty((N, T, D), dtype='f')
		dh, dc = 0, 0
		
		grads = [0,0,0]
		for t in reversed(range(T)):
			layer = self.layers[t]
			dx, dh, dc = layer.backward(dhs[:, t, :], dh, dc)
			try:
				dxs[:, t, :] = dx
			except:
				print(dxs.shape)
			for i , grad in enumerate(layer.grad):
				grads[i] += grad
				
		self.grad['Wx'] = grads[0]
		self.grad['Wh'] = grads[1]
		self.grad['b'] = grads[2]
		self.dh = dh
		
		return dxs
		
	def set_state(self, h, c=None):
		self.h, self.c= h,c
		
	def reset_state(self):
		self.h, self.c = None, None

	def save(self, name):
		params = {}
		for key, val in self.params.items():
			params[key] = val
		with open('./weight/new/TimeLSTM_'+name, 'wb') as f:
			pickle.dump(params, f)
	
	def load(self, name):
		with open('./weight/new/TimeLSTM_'+name, 'rb') as f:
			params = pickle.load(f)
		
		for key, val in params.items():
			try:
				self.params[key] = np.asarray(val)
			except:
				self.params[key] = cp.asnumpy(val)
	

class GRU:
	
	'''
	Gated Recurrent Unit
	'''
	
	def __init__(self,Wx,Wh,b):
		self.name = 'LSTM'
		#parameters
		self.params = [Wx, Wh, b]
		self.grad = [np.zeros_like(i) for i in self.params]
		
		#gate
		self.sigm1 = Sigmoid()
		self.sigm2 = Sigmoid()
		self.tanIn = Tanh()
		
		#data for backward
		self.In = None
		self.Hin = None
		self.To = None
		
		#other
		self.size = 0
		self.flops = 0
		
	def forward(self,input,Hin=None):
		Wx,Wh,b = self.params
		N, H = Hin.shape
		
		x = np.dot(input,Wx)
		h = np.dot(Hin,Wh)
		
		#slice
		r = x[:, :H]+h[:, :H]+b[:H]
		u = x[:, H:2*H]+h[:, H:2*H]+b[H:2*H]
		xo = x[:, 2*H:]+b[2*H:]
		ho = h[:, 2*H:]
		
		Sr = self.sigm1.forward(r)			#reset Gate(sigmoid)
		Su = self.sigm2.forward(u)			#update gate(sigmoid)
		To = self.tanIn.forward(ho*Sr+xo)	#Ouput gate(Tanh)
		Hout = (Su-1)*Hin+(Su*To)
		
		self.Hin = Hin
		self.In = input
		self.To = To

		return Hout
	
	def backward(self,dOut,dHout):
		Wx,Wh,b = self.params
		dHout = dOut+dHout
		
		dSu = dHout*self.Hin+dHout*self.To
		dTo = dHout*self.sigm2.out
		dSr = dTo*self.Hin
		dHin = dTo*self.sigm1.out
		
		dr = self.sigm1.backward(dSr)
		du = self.sigm1.backward(dSu)
		do = self.tanIn.backward(dTo)
		
		dx = np.hstack((dr,du,do))
		dh = np.hstack((dr,du,do*self.sigm1.out))
		
		self.grad[2] = np.hstack((np.sum(dr,axis=0),np.sum(du,axis=0),np.sum(do,axis=0)))
		self.grad[1] = np.dot(self.Hin.T, dh)
		self.grad[0] = np.dot(self.In.T, dx)

		dHin += np.dot(dh, Wh.T)
		dIn = np.dot(dx, Wx.T)
	
		return dIn,dHin


class TimeGRU:
	
	def __init__(self,stateful=False):
		self.name = 'TimeGRU'
		#initialize
		self.h = None
		self.dh = None
		self.stateful = stateful
		
		#parameters
		self.params={}
		self.params['Wx'] = None
		self.params['Wh'] = None
		self.params['b'] = None
		self.grad = {}
		self.grad['Wx'] = None
		self.grad['Wh'] = None
		self.grad['b'] = None	
		
		#other
		self.flops = 0
		self.size = 0
		self.optimizer = Adam(0.01)
		
	def forward(self, xs):
		Wx, Wh, b = self.params['Wx'],self.params['Wh'],self.params['b']
		N, T, D = xs.shape
		H = Wh.shape[0]
		
		self.layers = []
		hs = np.empty((N, T, H), dtype='f')
		
		if not self.stateful or self.h is None:
			self.h = np.zeros((N,H), dtype='f')

		for t in range(T):
			layer = GRU(Wx,Wh,b)
			self.h = layer.forward(xs[:, t, :], self.h)
			hs[:, t, :] = self.h
			
			self.layers.append(layer)
			
		return hs
		
	def backward(self,dhs):
		Wx, Wh, b = self.params['Wx'],self.params['Wh'],self.params['b']
		N, T, H = dhs.shape
		D = Wx.shape[0]
		
		dxs = np.empty((N, T, D), dtype='f')
		dh = 0
		
		grads = [0,0,0]
		for t in reversed(range(T)):
			layer = self.layers[t]
			dx, dh = layer.backward(dhs[:, t, :], dh)
			try:
				dxs[:, t, :] = dx
			except:
				print(dxs.shape)
			for i , grad in enumerate(layer.grad):
				grads[i] += grad
				
		self.grad['Wx'] = grads[0]
		self.grad['Wh'] = grads[1]
		self.grad['b'] = grads[2]
		self.dh = dh
		
		return dxs
		
	def set_state(self, h, c=None):
		self.h = h
		
	def reset_state(self):
		self.h = None

	def save(self, name):
		params = {}
		for key, val in self.params.items():
			params[key] = val
		with open('./weight/new/TimeGRU_'+name, 'wb') as f:
			pickle.dump(params, f)
	
	def load(self, name):
		with open('./weight/new/TimeGRU_'+name, 'rb') as f:
			params = pickle.load(f)
		
		for key, val in params.items():
			try:
				self.params[key] = np.asarray(val)
			except:
				self.params[key] = cp.asnumpy(val)



'''
Other Layer
'''


class SoftmaxWithLoss:
	
	'''
	Softmax layer+CorssEntropyError layer
	'''
	
	def __init__(self):
		self.name = 'Softmax'
		#Initialize
		self.params = {}
		self.shapeIn = None
		self.shapeOut = None
		self.loss = None
		self.y = None 
		self.t = None 
		self.size = 0
		self.flops = 0
		
		
	def process(self, x):
		y = softmax(x)

		return y
		
	def forward(self, x, t):
		self.t = t
		self.y = softmax(x)
		y = self.y
		self.loss = cross_entropy_error(self.y, self.t)
		loss = self.loss
		
		return loss
	
	def backward(self, dout=1):
		batch_size = self.t.shape[0]
		if self.t.size == self.y.size:
			dx = (self.y - self.t) / batch_size
		else:
			dx = self.y.copy()
			dx[np.arange(batch_size), self.t] -= 1
			dx = dx / batch_size
		
		return dx
		
