# coding: utf-8

#Original module
from functions import *				#im2col and col2im function(written by 斎藤康毅)
from optimizer import *				#SGD/Adam/optimizer/Momentum/RMSprop
from Activation import *			#ReLU/Elu/GELU/ISRL/ISRLU/Sigmoid/Softplus/Softsign/Tanh/Arctan
#from np.dot import np.dot

#Native Module
import pickle
import numpy as np
from numpy import exp


#
#Network
#

class Dense:
	
	'''
	Full Conected Layer
	'''
	
	def __init__(self, output_size, AF=Sigmoid,learning_rate=0.01,optimizer=Adam):
		self.name = 'Dense'
		#initialize
		self.output_size = output_size
		self.shapeIn = None
		self.shapeOut = None
		self.params = {}
		self.params['W1'] = None	#Weight
		self.params['b1'] = np.zeros(output_size)	#Bias
		self.size = None
		self.AF = AF()	#activation function
		self.x = None	#input
		self.grad = {}	#Gradient of weight and bias
		self.optimizer = optimizer(lr = learning_rate)	#Optimizer
	
	def forward(self, x):
		W1 = self.params['W1']		#load weight
		b1 = self.params['b1']		#load bias
		
		self.x = x
		a1 = np.dot(x, W1) + b1		#Sum of input
		z1 = self.AF.forward(a1)	#Activation
		
		return z1
	
	def backward(self,error):
		W1 = self.params['W1']
		b1 = self.params['b1']
		
		dy = error	
		da1 = self.AF.backward(dy)	#Backpropagation for Activation Function
		dx = np.dot(da1, W1.T)		#BP for input
		
		try:
			self.grad['W1'] = np.dot(self.x.T,da1)	#BP for weight
		except:
			self.grad['W1'] = np.dot(self.x.reshape(self.x.shape[0],self.x.shape[2]).T, da1.reshape(da1.shape[0],da1.shape[2]))
			self.grad['W1'] = self.grad['W1'].reshape(self.grad['W1'].size//da1.shape[2],da1.shape[2])
		
		self.grad['b1'] = np.sum(da1, axis=0)	#BP for bias

		return dx
	
	def save(self, name="Dense_W"):
		params = {}
		for key, val in self.params.items():
			params[key] = val
	
		with open('./weight/Dense_W_'+name, 'wb') as f:
			pickle.dump(params, f)
	
	def load(self, name="Dense_W"):
		with open('./weight/Dense_W_'+name, 'rb') as f:
			params = pickle.load(f)
		
		for key, val in params.items():
			if self.params[key].shape == val.shape:
				self.params[key] = val


class Conv:

	'''
	Convolution Layer
	'''
	
	def __init__(self,conv_param,init_std=1,AF=Elu(),learning_rate=0.01,optimizer=Adam):
		self.name = 'ConvNet'
		#Initialize
		self.shapeIn = None
		self.shapeOut = None
		self.f_num = conv_param['f_num']			#amount of filters
		self.f_size = conv_param['f_size']		#Size of filters
		self.f_pad = conv_param['pad']			#Padding size
		self.f_stride = conv_param['stride']		#Step of filters
		
		self.params = {}
		self.params['W1'] = None
		self.params['b1'] = np.zeros(self.f_num)
		self.size = None
		self.grad = {}
		self.stride = self.f_stride
		self.pad = self.f_pad
		
		self.AF = AF
		self.optimizer = optimizer(lr = learning_rate)
		
		self.x = None   
		self.col = None
		self.col_W = None
		
	def forward(self, x):
		FN, C, FH, FW = self.params['W1'].shape
		N, C, H, W = x.shape
		
		out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
		out_w = 1 + int((W + 2*self.pad - FW) / self.stride)
		
		col = im2col(x, FH, FW, self.stride, self.pad)
		col_W = self.params['W1'].reshape(FN, -1).T
		out = np.dot(col, col_W) + self.params['b1']
		out = self.AF.forward(out)
		out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
		self.x = x
		self.col = col
		self.col_W = col_W

		return out
	
	def backward(self, dout):
		FN, C, FH, FW = self.params['W1'].shape
		dout = dout.transpose(0,2,3,1).reshape(-1, FN)
		dout = self.AF.backward(dout)
		self.grad['b1'] = np.sum(dout, axis=0)
		self.grad['W1'] = np.dot(self.col.T, dout)
		self.grad['W1'] = self.grad['W1'].transpose(1, 0).reshape(FN, C, FH, FW)
		dcol = np.dot(dout, self.col_W.T)
		dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

		return dx
	
	def save(self, name="Conv_W"):
		params = {}
		for key, val in self.params.items():
			params[key] = val
	
		with open('./weight/Conv_W_'+name, 'wb') as f:
			pickle.dump(params, f)
	
	def load(self, name="Conv_W"):
		with open('./weight/Conv_W_'+name, 'rb') as f:
			params = pickle.load(f)
		
		for key, val in params.items():
			if self.params[key].shape == val.shape:
				self.params[key] = val


class DeConv:

	'''
	Transpose Convolution Layer
	'''
	
	def __init__(self,conv_param,init_std=1,AF=Elu(),learning_rate=0.01,optimizer=Adam):
		self.name = 'DeConvNet'
		#Initialize
		self.shapeIn = None
		self.shapeOut = None
		self.f_num = conv_param['f_num']
		self.f_size = conv_param['f_size']
		self.f_stride = conv_param['stride']
		
		self.params = {}
		self.params['W1'] = None
		self.size = None
		self.grad = {}
		self.stride = self.f_stride
		
		self.AF = AF
		self.optimizer = optimizer(lr = learning_rate)
		
		self.x = None   
		self.col = None
		self.col_W = None
			
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
		self.x = x
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
		dx = dcol.reshape((self.x.shape[0],self.x.shape[2],self.x.shape[3],-1)).transpose(0,3,1,2)

		return dx
	
	def save(self, name="Conv_W"):
		params = {}
		for key, val in self.params.items():
			params[key] = val
	
		with open('./weight/Conv_W_'+name, 'wb') as f:
			pickle.dump(params, f)
	
	def load(self, name="Conv_W"):
		with open('./weight/Conv_W_'+name, 'rb') as f:
			params = pickle.load(f)
		
		for key, val in params.items():
			if self.params[key].shape == val.shape:
				self.params[key] = val


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
		self.pool_h = pool_h
		self.pool_w = pool_w
		self.stride = stride
		self.pad = pad
		self.size = 0
		self.x = None
		self.arg_max = None
		self.out_size = None
	
	def forward(self, x):
		N, C, H, W = x.shape
		out_h = int(1 + (H - self.pool_h) / self.stride)
		out_w = int(1 + (W - self.pool_w) / self.stride)
		self.out_size = (C,out_h,out_w)
		
		col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
		col = col.reshape(-1, self.pool_h*self.pool_w)
		arg_max = np.argmax(col, axis=1)
		out = np.max(col, axis=1)
		out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
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


class Flatten:
	
	'''
	3D(2D)data to 1D
	'''
	
	def __init__(self):
		self.name='Flatten'
		#Initialize
		self.shapeIn = None
		self.shapeOut = None
		self.in_size=None
		self.size = 0
		
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
		self.shapeIn = None
		self.shapeOut = None
		self.params = {}
		self.params['gamma']= gamma
		self.params['beta'] = beta
		self.momentum = momentum
		self.input_shape = None # Conv is 4d(N C H W), FCN is 2d(N D)

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
		
		self.size = 2
		
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
	
		with open('./weight/Batch_Norm_'+name, 'wb') as f:
			pickle.dump(params, f)
	
	def load(self, name="Batch_Norm"):
		with open('./weight/Batch_Norm_'+name, 'rb') as f:
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
		self.layer = layer
		self.AF = None
		self.layers = []
		
		self.size = 0
		self.shapeOut = None
		self.shapeIn = None
		
	def initial(self,data,init_std,rate=0.001,AF=Elu,optimizer=Adam):
		Ini = True
		init = data
		self.AF = AF()
		
		for i in range(len(self.layer)):
				
			if self.layer[i].name == 'ConvNet':
				self.layer[i].params['W1'] = init_std * np.random.randn(self.layer[i].f_num, init.shape[1], self.layer[i].f_size, self.layer[i].f_size)
				self.layer[i].size = self.layer[i].params['W1'].size+self.layer[i].params['b1'].size
				
				self.size += self.layer[i].size
				init = self.layer[i].forward(init)
			
			elif self.layer[i].name == 'Dense':
				if init.ndim!=2:
					init = init.reshape(init.shape[0],-1)
				self.layer[i].params['W1'] = init_std * np.random.randn(init.size, self.layer[i].output_size)
				self.layer[i].size = self.layer[i].params['W1'].size+self.layer[i].params['b1'].size
				
				self.size += self.layer[i].size
				init = self.layer[i].forward(init)
		
		for i in self.layer:
			i.AF = ID()
			i.optimizer = optimizer(rate)
			self.layers.append(i)
			self.layers.append(BatchNorm())
			self.layers.append(AF())
		
		self.layers.pop()
		
		return init
	
	def forward(self,x):
		out = x
		for i in self.layers:
			out = i.forward(out)
		
		return self.AF.forward(out)+x
	
	def backward(self,dout):
		dx = dout
		dx = self.AF.backward(dx)
		self.layers.reverse()
		for i in self.layers:
			dx = i.backward(dx)
		
		self.layers.reverse()
		
		return dx+dout
	
	def train(self):
		for i in self.layers:
			try:
				i.optimizer.update(i.params,i.grad)
		
			except AttributeError:
				pass
	
	def save(self, name="Res"):
		j = 1
		for i in self.layers:
			try:
				i.save(str(j)+'_Res_'+name)
			
			except AttributeError: #AF pooling Flatten
				pass
			
			j+=1
		
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


class ResLayerV2:
	
	def __init__(self,layer):
		self.name = 'ResLayer'
		self.layer = layer
		self.AF = None
		self.layers = []
		
		self.size = 0
		self.shapeOut = None
		self.shapeIn = None
		
	def initial(self,data,init_std,rate=0.001,AF=Elu,optimizer=Adam):
		Ini = True
		init = data
		self.AF = AF()
		
		for i in range(len(self.layer)):
				
			if self.layer[i].name == 'ConvNet':
				self.layer[i].params['W1'] = init_std * np.random.randn(self.layer[i].f_num, init.shape[1], self.layer[i].f_size, self.layer[i].f_size)
				self.layer[i].size = self.layer[i].params['W1'].size+self.layer[i].params['b1'].size
				
				self.size += self.layer[i].size
				init = self.layer[i].forward(init)
			
			elif self.layer[i].name == 'Dense':
				if init.ndim!=2:
					init = init.reshape(init.shape[0],-1)
				self.layer[i].params['W1'] = init_std * np.random.randn(init.size, self.layer[i].output_size)
				self.layer[i].size = self.layer[i].params['W1'].size+self.layer[i].params['b1'].size
				
				self.size += self.layer[i].size
				init = self.layer[i].forward(init)
		
		for i in self.layer:
			i.AF = ID()
			i.optimizer = optimizer(rate)
			self.layers.append(BatchNorm())	
			self.layers.append(AF())
			self.layers.append(i)	
		
		return init
	
	def forward(self,x):
		out = x
		for i in self.layers:
			out = i.forward(out)
		
		return out+x
	
	def backward(self,dout):
		dx = dout
		self.layers.reverse()
		for i in self.layers:
			dx = i.backward(dx)
		
		self.layers.reverse()
		
		return dx+dout
	
	def train(self):
		for i in self.layers:
			try:
				i.optimizer.update(i.params,i.grad)
		
			except AttributeError:
				pass
	
	def save(self, name="Res"):
		j = 1
		for i in self.layers:
			try:
				i.save(str(j)+'_Res_'+name)
			
			except AttributeError: #AF pooling Flatten
				pass
			
			j+=1
		
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


class SoftmaxWithLoss:
	
	'''
	Softmax layer+CorssEntropyError layer
	'''
	
	def __init__(self):
		self.name = 'Softmax'
		#Initialize
		self.shapeIn = None
		self.shapeOut = None
		self.loss = None
		self.y = None 
		self.t = None 
		self.size = 0
	
	def forward(self, x, t):
		self.t = t
		self.y = softmax(x)
		y = self.y
		self.loss = cross_entropy_error(self.y, self.t)
		loss = self.loss
		
		return loss
		
	def forward_without_loss(self, x):
		y = softmax(x)

		return y
	
	def backward(self, dout=1):
		batch_size = self.t.shape[0]
		if self.t.size == self.y.size:
			dx = (self.y - self.t) / batch_size
		else:
			dx = self.y.copy()
			dx[np.arange(batch_size), self.t] -= 1
			dx = dx / batch_size
		
		return dx