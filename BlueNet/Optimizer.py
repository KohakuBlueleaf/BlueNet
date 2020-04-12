# coding: utf-8
import sys
from BlueNet.setting import _np

def L1_Norm(weight,grad,decay_rate=0.00001):
	decay = _np.sum(_np.abs(weight))
	grad += decay*decay_rate
	
	return grad


def L2_Norm(weight,grad,decay_rate=0.00001):
	decay = _np.sum(weight**2)**0.5
	grad -= decay*decay_rate
	
	return grad


class SGD:

	"""Stochastic Gradient Descent"""

	def __init__(self, lr=0.01):
		self.lr = lr
		self.normalization = normalization
		
	def update(self, params, grads):
		for key in params.keys():
			params[key] -= self.lr * grads[key] 


class Momentum:

	"""Momentum SGD"""

	def __init__(self, lr=0.01, momentum=0.9):
		self.lr = lr
		self.momentum = momentum
		self.v = None
		self.normalization = normalization
		
	def update(self, params, grads):
		if self.v is None:
			self.v = {}
			for key, val in params.items():								
				self.v[key] = _np.zeros_like(val)
				
		for key in params.keys():
			self.v[key] = self.momentum*self.v[key] - self.lr*grads[key] 
			params[key] += self.v[key]


class Nesterov:

	"""Nesterov's Accelerated Gradient (http://arxiv.org/abs/1212.0901)"""

	def __init__(self, lr=0.01, momentum=0.9):
		self.lr = lr
		self.momentum = momentum
		self.v = None
		self.normalization = normalization
		
	def update(self, params, grads):
		if self.v is None:
			self.v = {}
			for key, val in params.items():
				self.v[key] = _np.zeros_like(val)
			
		for key in params.keys():
			self.v[key] *= self.momentum
			self.v[key] -= self.lr * grads[key]
			params[key] += self.momentum * self.momentum * self.v[key]
			params[key] -= (1 + self.momentum) * self.lr * grads[key]


class AdaGrad:

	"""AdaGrad"""

	def __init__(self, lr=0.01):
		self.lr = lr
		self.h = None
		self.normalization = normalization
		
	def update(self, params, grads):
		if self.h is None:
			self.h = {}
			for key, val in params.items():
				self.h[key] = _np.zeros_like(val)
			
		for key in params.keys():
			self.h[key] += grads[key] * grads[key]
			params[key] -= self.lr * grads[key] / (_np.sqrt(self.h[key]) + 1e-7)


class RMSprop:

	"""RMSprop"""

	def __init__(self, lr=0.01, decay_rate = 0.99):
		self.lr = lr
		self.decay_rate = decay_rate
		self.h = None
		self.normalization = normalization
		
	def update(self, params, grads):
		if self.h is None:
			self.h = {}
			for key, val in params.items():
				self.h[key] = _np.zeros_like(val)
			
		for key in params.keys():
			self.h[key] *= self.decay_rate
			self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]
			params[key] -= self.lr * grads[key] / (_np.sqrt(self.h[key]) + 1e-7)


class Adam:

	"""Adam (http://arxiv.org/abs/1412.6980v8)"""

	def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
		self.lr = lr
		self.beta1 = beta1
		self.beta2 = beta2
		self.iter = 0
		self.m = {}
		self.v = {}
		
	def update(self, params, grads):
		if self.m == {}:
			for key, val in params.items():
				self.m[key] = _np.zeros_like(val)
				self.v[key] = _np.zeros_like(val)
		
		self.iter += 1
		lr_t  = self.lr * _np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)		 
				
		for key in params.keys():
			try:
				self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
				self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
				
				params[key] -= lr_t * self.m[key] / (_np.sqrt(self.v[key]) + 1e-7)
			except ValueError:
				pass

