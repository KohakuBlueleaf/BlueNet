# coding: utf-8
import pickle
from random import randint as rand
import numpy as np
import gzip

img_size = 784

dataset = {}
testset = {}

file = {
	'train_img':'./gzip/emnist-letters-train-images-idx3-ubyte.gz',
	'train_label':'./gzip/emnist-letters-train-labels-idx1-ubyte.gz',
	'test_img':'./gzip/emnist-letters-test-images-idx3-ubyte.gz',
	'test_label':'./gzip/emnist-letters-test-labels-idx1-ubyte.gz'
}

def _change_one_hot_label(X):
	T = np.zeros((len(X), 26))
	for idx, row in enumerate(T):
		row[X[idx]-1] = 1

	return T

def load_labels(file):
	with gzip.open(file, 'r',) as f:
		labels = np.frombuffer(f.read(), np.uint8, offset=8)
	
	return labels

def load_imgs(file):
	with gzip.open(file, 'r',) as f:
		imgs = np.frombuffer(f.read(), np.uint8, offset=16)
	imgs = imgs.reshape(-1, img_size)
	return imgs

def load_emnist(normalize=True, flatten=True, one_hot_label=True):
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
		dataset['labels'] = _change_one_hot_label(dataset['labels'])
		testset['labels'] = _change_one_hot_label(testset['labels'])

	if not flatten:
		dataset['data'] = dataset['data'].reshape(-1, 1, 28, 28)
		testset['data'] = testset['data'].reshape(-1, 1, 28, 28)

	return (dataset['data'], dataset['labels']), (testset['data'], testset['labels'])

