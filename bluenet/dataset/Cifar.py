# coding: utf-8
import sys,os
sys.path.append("..") 
import pickle
import numpy as np
from PIL import Image
from bluenet.functions import _change_one_hot_label,label_smoothing


dataset_dir = os.path.dirname(os.path.abspath(__file__))
file = dataset_dir+'/data/cifar_data/cifar_data_batch_'

def load_cifar(normalize=True, flatten=True, one_hot_label=False, smooth=False, type=np.float32):
	dataset = {}
	with open(file+'01', 'rb') as f:
		dataset1 = pickle.load(f)
	
	with open(file+'02', 'rb') as f:
		dataset2 = pickle.load(f)
	
	with open(dataset_dir+'/data/cifar_data/cifar_test_batch', 'rb') as f:
		testset = pickle.load(f,encoding='bytes')
	
	dataset[b'data'] = np.asarray(np.vstack((dataset1[b'data'],dataset2[b'data'])))
	dataset[b'labels'] = np.asarray(np.hstack((dataset1[b'labels'],dataset2[b'labels'])))
	testset[b'data'] = np.asarray(testset[b'data'])
	testset[b'labels'] = np.asarray(testset[b'labels'])
	
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
	(A,B),(C,D) = load_cifar()
	print(A.shape)
	print(B.shape)
	print(C.shape)
	print(D.shape)
	