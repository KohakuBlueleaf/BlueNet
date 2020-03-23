# coding: utf-8
import sys
sys.path.append("..") 
import pickle
from random import randint as rand
from BlueNet.functions import _change_one_hot_label,label_smoothing
import numpy as np
from PIL import Image


file = 'BlueNet/database/cifar_data/data_batch_'

def load_cifar(normalize=True, flatten=True, one_hot_label=False, smooth=False, type=np.float32):
	with open(file+str(1), 'rb',) as f:
		dataset1 = pickle.load(f,encoding='bytes')
	with open(file+str(2), 'rb',) as f:
		dataset2 = pickle.load(f,encoding='bytes')
	with open(file+str(3), 'rb',) as f:
		dataset3 = pickle.load(f,encoding='bytes')
	with open(file+str(4), 'rb',) as f:
		dataset4 = pickle.load(f,encoding='bytes')
	with open(file+str(5), 'rb',) as f:
		dataset5 = pickle.load(f,encoding='bytes')
	dataset = {}
	with open('./database/cifar_data/test_batch', 'rb') as f:
		testset = pickle.load(f,encoding='bytes')
	
	dataset[b'data'] = np.vstack((dataset1[b'data'],dataset2[b'data'],dataset3[b'data'],dataset4[b'data'],dataset5[b'data']))
	dataset[b'labels'] = np.hstack((dataset1[b'labels'],dataset2[b'labels'],dataset3[b'labels'],dataset4[b'labels'],dataset5[b'labels']))
	
	
	if normalize:
		dataset[b'data'] = dataset[b'data'].astype(type)
		dataset[b'data'] /= 255.0
		testset[b'data'] = testset[b'data'].astype(type)
		testset[b'data'] /= 255.0

	if not flatten:
		#dataset[b'data'] = np.vstack((dataset[b'data'].reshape(50000, 3, 32, 32),np.flip(dataset[b'data'].reshape(50000, 3, 32, 32),(1,3))))
		#dataset[b'labels'] = np.hstack((dataset[b'labels'],dataset[b'labels']))
		dataset[b'data'] = dataset[b'data'].reshape(50000, 3, 32, 32)
		testset[b'data'] = testset[b'data'].reshape(10000, 3, 32, 32)
	
	if one_hot_label:
		dataset[b'labels'] = _change_one_hot_label(dataset[b'labels'],10)
		testset[b'labels'] = _change_one_hot_label(testset[b'labels'],10)
	
	if smooth:
		dataset[b'labels'] = label_smoothing(dataset[b'labels'],0.1)
		testset[b'labels'] = label_smoothing(testset[b'labels'],0.1)
	
	return (dataset[b'data'], dataset[b'labels'].astype(type)), (testset[b'data'], testset[b'labels'].astype(type))

if __name__ == '__main__':
	(a,b),(c,d) = load_cifar(False,False,True)
	print(a.shape,b.shape)