# coding: utf-8
import pickle
from random import randint as rand
import numpy as np
def _change_one_hot_label(X):
	T = np.zeros((len(X), 10))
	for idx, row in enumerate(T):
		row[X[idx]] = 1

	return T
file = './cifar/data_batch_'
def load_cifar(normalize=True, flatten=True, one_hot_label=False):
	with open(file+str(rand(1,5)), 'rb',) as f:
		dataset = pickle.load(f,encoding='bytes')
		#print (dataset)
	with open('./cifar/test_batch', 'rb') as f:
		testset = pickle.load(f,encoding='bytes')
	
	if normalize:
		dataset[b'data'] = dataset[b'data'].astype(np.float32)
		dataset[b'data'] /= 255.0
		testset[b'data'] = testset[b'data'].astype(np.float32)
		testset[b'data'] /= 255.0

	if one_hot_label:
		dataset[b'labels'] = _change_one_hot_label(dataset[b'labels'])
		testset[b'labels'] = _change_one_hot_label(testset[b'labels'])

	if not flatten:
		dataset[b'data'] = dataset[b'data'].reshape(10000, 3, 32, 32)
		testset[b'data'] = testset[b'data'].reshape(10000, 3, 32, 32)

	return (dataset[b'data'], dataset[b'labels']), (testset[b'data'], testset[b'labels'])