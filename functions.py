# coding: utf-8
import numpy as np
import math
from scipy.special import erf

'''
Activation Functions
'''

sqrt2 = 2**0.5

def identity_function(x):
	
	return x

def step_function(x):
	
	return np.array(x > 0, dtype=np.int)

def sigmoid(x):
	
	return 1 / (1 + np.exp(-x))	
	
def sigmoid_grad(x):
	
	return (1.0 - sigmoid(x)) * sigmoid(x)

def softplus(x):
	
	return np.log(1+np.exp(x))

def softplus_grad(x):
	
	return sigmoid(x)

def elu(x):
	
	return np.exp(x)-1
	
def elu_grad(x):

	return np.exp(np.exp(x)-1)+1

def sinh(x):
	
	return ((np.exp(x) - np.exp(-x)) / 2)

def cosh(x):
	
	return ((np.exp(x) + np.exp(-x)) / 2)

def tanh(x):
	
	return sinh(x) / cosh(x)

def sech(x):
	
	return 1/cosh(x)

def tanh_grad(x):
	
	return 1-(tanh(x)**2)

def relu(x):
	
	return np.maximum(0, x)

def relu_grad(x):
	grad = np.zeros(x)
	grad[x>=0] = 1
	
	return grad
	
def arctan(x):

	return(np.arctan(x))
	
def arctan_grad(x):

	return(1/(x**2+1))

def softsign(x):
	
	return x/(1+np.abs(x))
	
def softsign_grad(x):
	
	return 1/(1+np.abs(x))**2
	
def isru(x,a=1):

	return x*(1/(1+a*x**2)**0.5)

def isru_grad(x,a=1):

	return (1/(1+a*x**2)**0.5)**3
	
def isru_bound(a):
	
	return (1/a**0.5),(-1/a**0.5)

def erf_grad(x):
	
	return (2/np.pi**0.5)*(np.exp(-x**2))

def gelu_erf(x): #use scipy.special.erf(also approximation but more accurate)
	
	return 0.5*x*(1+erf(x/sqrt2))

def gelu_erf_grad(x):
	
	return 0.25*(sqrt2*x*erf_grad(x/sqrt2)+2*erf(x/sqrt2)+2)


'''
Other Functions
'''

def mean_squared_error(y, t):
	
	return 0.5 * np.sum((y-t)**2)

def cross_entropy_error(y, t):
	if y.ndim == 1:
		t = t.reshape(1, t.size)
		y = y.reshape(1, y.size)
		
	if t.size == y.size:
		t = t.argmax(axis=1)
			 
	batch_size = y.shape[0]
	
	return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

def softmax(x):
	if x.ndim == 2:
		x = x.T
		x = x - np.max(x, axis=0)
		x = np.exp(x)
		y = x / np.sum(x, axis=0)
	
		return y.T 
	
	x = x - np.max(x)
	x = np.exp(x)
	y = x / np.sum(x)
	
	return y

def softmax_loss(X, t):
	y = softmax(X)
	
	return cross_entropy_error(y, t)

def numerical_gradient_n(f, x):
	h = 1e-4 # 0.0001
	
	a = x+h
	b = x-h
	A = f(a)
	B = f(b)
	grad = (A-B)/(h*2)
	
	return grad
	
def numerical_gradient_1d(f, x):
	h = 1e-4 # 0.0001
	grad = np.zeros_like(x)
	
	for idx in range(x.size):
		tmp_val = x[idx]
		x[idx] = float(tmp_val) + h
		fxh1 = f(x) # f(x+h)
		
		x[idx] = tmp_val - h 
		fxh2 = f(x) # f(x-h)
		grad[idx] = (fxh1 - fxh2) / (2*h)
		
		x[idx] = tmp_val # 値を元に戻す
		
	return grad

def numerical_gradient_2d(f, X):
	if X.ndim == 1:
		return _numerical_gradient_1d(f, X)
	else:
		grad = np.zeros_like(X)
		
		for idx, x in enumerate(X):
			grad[idx] = _numerical_gradient_1d(f, x)
		
		return grad

def numerical_gradient(f, x):
	h = 1e-4 # 0.0001
	grad = np.zeros_like(x)
	
	it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
	while not it.finished:
		idx = it.multi_index
		tmp_val = x[idx]
		x[idx] = float(tmp_val) + h
		fxh1 = f(x) # f(x+h)
		
		x[idx] = tmp_val - h 
		fxh2 = f(x) # f(x-h)
		grad[idx] = (fxh1 - fxh2) / (2*h)
		
		x[idx] = tmp_val # 値を元に戻す
		it.iternext()   
		
	return grad

def change_one_hot_label(X):
	T = np.zeros((X.size, 10))
	for idx, row in enumerate(T):
		row[X[idx]] = 1

	return T
	

'''
im2col col2im
'''

def smooth_curve(x):
	window_len = 11
	s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
	w = np.kaiser(window_len, 2)
	y = np.convolve(w/w.sum(), s, mode='valid')
	return y[5:len(y)-5]

def shuffle_dataset(x, t):
	permutation = np.random.permutation(x.shape[0])
	x = x[permutation,:] if x.ndim == 2 else x[permutation,:,:,:]
	t = t[permutation]

	return x, t

def conv_output_size(input_size, filter_size, stride=1, pad=0):
	return (input_size + 2*pad - filter_size) / stride + 1

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
	N, C, H, W = input_data.shape
	out_h = (H + 2*pad - filter_h)//stride + 1
	out_w = (W + 2*pad - filter_w)//stride + 1

	img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
	col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

	for y in range(filter_h):
		y_max = y + stride*out_h
		for x in range(filter_w):
			x_max = x + stride*out_w
			col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

	col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
	return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
	N, C, H, W = input_shape
	out_h = (H + 2*pad - filter_h)//stride + 1
	out_w = (W + 2*pad - filter_w)//stride + 1
	col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

	img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
	for y in range(filter_h):
		y_max = y + stride*out_h
		for x in range(filter_w):
			x_max = x + stride*out_w
			img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

	return img[:, :, pad:H + pad, pad:W + pad]