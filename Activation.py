# coding: utf-8
from functions import *	
from setting import exp			


'''
Active Function
'''

class ID:
	
	'''
	Identify y = x
	'''
	def __init__(self):
		self.name='ID'
	
	def forward(self, x):
		
		return x
	
	def backward(self, dout):

		return dout


class Relu:
	
	'''
	整流線性單元(Rectified Linear Unit)
	'''
	
	def __init__(self):
		self.mask = None 		#mask for x<=0
		self.name='Relu'
	
	def forward(self, x):
		self.mask = (x <= 0)
		out = x.copy() 
		out[self.mask] = 0 		#if x<=0 →→ x=0

		return out
	
	def backward(self, dout):
		dout[self.mask] = 0 	#if outx=0 dx=0
		dx = dout

		return dx
	
	
class Leaky:
	
	'''
	Leaky ReLU
	'''
	
	def __init__(self):
		self.mask = None 			#mask for x<=0
		self.name='LReLU'
	
	def forward(self, x):
		self.mask = (x <= 0)
		out = x.copy() 
		out[self.mask] *= 0.01 		#if x<=0 →→ x=a*x

		return out
	
	def backward(self, dout):
		dout[self.mask] *= 0.01 	#if outx=0 dx=a*dx
		dx = dout

		return dx


class Elu:
	
	'''
	Exponential Linear Unit

	If x>0 out = x
	If x<=0 out = e^x-1 (Gradient = e^(e^x-1)+1
	'''
	
	def __init__(self):
		self.mask = None											#mask for x<=0
		self.out = None												#Store the output(for backpropagation)
		self.name='Elu'
	
	def forward(self, x):
		self.mask = (x <= 0)	
		out = x.copy()
		out[self.mask] = exp(out[self.mask])-1						#if x<0→→out = e^x-1
		self.out = out
		
		return out
	
	def backward(self, dout):
		dout[self.mask] = dout[self.mask]*(self.out[self.mask]+1)	#f'(x)=f(x)+1
		self.mask = None
		self.out = None
		dx = dout

		return dx


class GELU:
	
	'''
	Gaussian Error Linear Units
	Definition is 0.5x(1+erf(x/√2)) 
	We use scipy.special.erf/cupyx.scipy.special.erf in here
	The Derivative function of GELU is compute by WolframAlpha
	(See functions.py)
	'''
	
	def __init__(self):
		self.name='GELU'
		self.In = None
	
	def forward(self,x):
		self.In = x
		out = gelu_erf(x)
		
		return out
		
	def backward(self,dout):
		dx = dout*gelu_erf_grad(self.In)
		self.In = None
		
		return dx


class ISRLU:
	
	'''
	INVERSE SQUARE ROOT LINEAR UNIT
	'''
	
	def __init__(self, a=1):
		self.name='ISRLU'
		self.In = None
		self.mask = None
		self.a = a
	
	def forward(self, x):
		self.mask = (x<=0)
		self.In = x
		out = x.copy()
		out[self.mask] = isru(out[self.mask],self.a)
		
		return out
		
	def backward(self, dout):
		dout[self.mask] = dout[self.mask]*isru_grad(self.In[self.mask],self.a)
		dx = dout
		
		return dx


class ISRU:
	
	'''
	INVERSE SQUARE ROOT UNIT
	'''
	
	def __init__(self, a = 1):
		self.name='ISRU'
		self.In = None
		self.a = a
	
	def forward(self, x):
		self.In = x
		out = isru(x,self.a)
		
		return out
		
	def backward(self, dout):
		dx = dout * isru_grad(self.In,self.a)
		
		return dx


class Sigmoid:
	
	'''
	Sigmoid Fuction
	'''
	
	def __init__(self):
		self.name= 'Sigmoid'
		self.out = None								#Store the output(for backpropagation)
		self.flops = 0
		self.size = 0
	
	def forward(self, x):
		out = sigmoid(x)							#See functions.py
		self.out = out
	
		return out
	
	def backward(self, dout):
		dx = dout * (1.0 - self.out) * self.out		#f'(x) = f(x)*(1-f(x))

		return dx


class Softplus:
	
	'''
	Softplus Function
	'''
	
	def __init__(self):
		self.name= 'Softplus'
		self.In = None						#Store the input(for backpropagation)
	
	def forward(self, x):
		self.In = x
		out = softplus(x)					#see functions.py
	
		return out
	
	def backward(self, dout):
		dx = dout * softplus_grad(self.In)	#see functions.py

		return dx


class Softsign:
	
	'''
	Soft sign function
	'''
	
	def __init__(self):
		self.In = None
		self.name='Softsign'
	
	def forward(self, x):
		self.In = x
		out = softsign(self.In)
		
		return out
		
	def backward(self, dout):
		dx = dout * softsign_grad(self.In)
		
		return dx


class Softclip:
	
	'''
	Soft-clipping function
	'''
	
	def __init__(self):
		self.In = None
		self.name='Softclip'
	
	def forward(self, x):
		self.In = x
		out = softclip(self.In)
		
		return out
		
	def backward(self, dout):
		dx = dout * softclip_grad(self.In)
		
		return dx


class SQNL:

	'''
	Square Non-Linearity
	'''
	
	def __init__(self):
		self.name= 'SQNL'
		self.mask1 = None 					#mask for 2<x
		self.mask2 = None 					#mask for 0<=x<=2
		self.mask3 = None 					#mask for -2<=x<0
		self.mask4 = None 					#mask for x<-2
		self.IN = None 
	
	def forward(self, x):
		self.IN = x
		self.mask1 = (2<=x) 				#mask for 2<x
		self.mask2 = (0<=x)					#mask for 0<=x<=2
		self.mask3 = (x<0) 					#mask for -2<=x<0
		self.mask4 = (x<-2) 				#mask for x<-2
		
		out = x.copy()
		out[self.mask2]=out[self.mask2]-((out[self.mask2]**2)/4)
		out[self.mask1]=1
		out[self.mask3]=out[self.mask3]+((out[self.mask3]**2)/4)
		out[self.mask4]=-1
		
		return out
	
	def backward(self, dout):
		dout[self.mask2] *= (1-(self.IN[self.mask])/2)
		dout[self.mask1] = 0
		dout[self.mask3] *= (1+(self.IN[self.mask])/2)
		dout[self.mask4] = 0 
		dx = dout

		return dx


class Tanh:
	
	'''
	One of Hyperbolic functions. Upper bound is 1, Lower Bound is -1.
	'''
	
	def __init__(self):
		self.In = None 					#Store the input(for backpropagation)
	
	def forward(self, x):
		self.In = x
		out = tanh(x)					#see functions.py
	
		return out
	
	def backward(self, dout):
		dx = dout * tanh_grad(self.In)	#see functions.py

		return dx


class Arctan:

	'''
	One of Inverse trigonometric functions. Upper bound is π/2, Lower bound is -π/2
	'''

	def __init__(self):
		self.In = None
		self.name = 'ArcTan'
		self.size = 0
		self.flops = 0
	
	def forward(self, x):
		self.In = x
		out = arctan(x)
		
		return out
		
	def backward(self, dout):
		dx = dout * arctan_grad(self.In)
		
		return dx

