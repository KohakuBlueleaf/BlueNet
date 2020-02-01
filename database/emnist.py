# coding: utf-8
import sys
sys.path.append("..") 
import pickle
from random import randint as rand
import numpy as np
import gzip
from functions import _change_one_hot_label, label_smoothing

img_size = 784
dataset = {}
testset = {}
file = {
	'train_img':'database/gzip/emnist-letters-train-images-idx3-ubyte.gz',
	'train_label':'database/gzip/emnist-letters-train-labels-idx1-ubyte.gz',
	'test_img':'database/gzip/emnist-letters-test-images-idx3-ubyte.gz',
	'test_label':'database/gzip/emnist-letters-test-labels-idx1-ubyte.gz'
}

def load_labels(file):
	with gzip.open(file, 'r',) as f:
		labels = np.frombuffer(f.read(), np.uint8, offset=8)
	
	return labels

def load_imgs(file):
	with gzip.open(file, 'r',) as f:
		imgs = np.frombuffer(f.read(), np.uint8, offset=16)
	imgs = imgs.reshape(-1, img_size)
	return imgs

def load_emnist(normalize=True, flatten=True, one_hot_label=True, smooth=False, choose = 0):
	dataset['data'] = load_imgs(file['train_img'])
	testset['data'] = load_imgs(file['test_img'])
	dataset['labels'] = load_labels(file['train_label'])
	testset['labels'] = load_labels(file['test_label'])

	if normalize:
		dataset['data'] = dataset['data'].astype(np.float32)
		dataset['data'] /= 255.0
		testset['data'] = testset['data'].astype(np.float32)
		testset['data'] /= 255.0

	if one_hot_label:
		dataset['labels'] = _change_one_hot_label(dataset['labels']-1,26)
		testset['labels'] = _change_one_hot_label(testset['labels']-1,26)

	if not flatten:
		dataset['data'] = dataset['data'].reshape(-1, 1, 28, 28)
		testset['data'] = testset['data'].reshape(-1, 1, 28, 28)
	
	if choose == 0:
		data_choose = np.arange(0,len(dataset['data']))
		test_choose = np.arange(0,len(testset['data']))
	elif choose == 1:
		data_choose = np.arange(1,len(dataset['data'])+1,2)
		test_choose = np.arange(1,len(testset['data'])+1,2)
	elif choose == 2:
		data_choose = np.arange(0,len(dataset['data']),2)
		test_choose = np.arange(0,len(testset['data']),2)
	elif choose == 3:
		data_choose = np.arange(0,len(dataset['data']),3)
		test_choose = np.arange(0,len(testset['data']),3)
	elif choose == 4:
		data_choose = np.arange(1,len(dataset['data']+1),3)
		test_choose = np.arange(1,len(testset['data']+1),3)
	elif choose == 5:
		data_choose = np.arange(2,len(dataset['data'])+2,3)
		test_choose = np.arange(2,len(testset['data'])+2,3)
	
	if smooth:
		dataset['labels'] = label_smoothing(dataset['labels'],0.1)
		testset['labels'] = label_smoothing(testset['labels'],0.1)
	
	return (dataset['data'][data_choose], dataset['labels'][data_choose]), (testset['data'][test_choose], testset['labels'][test_choose])


if __name__ == '__main__':
	
	(x_train, t_train), (x_test, t_test) = load_emnist(flatten=False, one_hot_label=False)
	print(x_train.shape)
	print(t_train.shape)
	print(x_test.shape)
	print(t_test.shape)
