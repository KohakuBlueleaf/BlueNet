# coding: utf-8
import sys,os
sys.path.append("..")
import pickle
import numpy as np
from PIL import Image
from bluenet.functions import _change_one_hot_label,label_smoothing


dataset ={}
dataset1={}
dataset2={}

dataset_dir = os.path.dirname(os.path.abspath(__file__))

def load_cifar100(normalize=True, flatten=True, one_hot_label=False, smooth=False, type=np.float32):
	with open(dataset_dir+'/data/cifar100_data/cifar100_data_train1', 'rb',) as f:
		dataset1 = pickle.load(f,encoding='bytes')
	
	with open(dataset_dir+'/data/cifar100_data/cifar100_data_train2', 'rb',) as f:
		dataset2 = pickle.load(f,encoding='bytes')
	
	with open(dataset_dir+'/data/cifar100_data/cifar100_data_test', 'rb') as f:
		testset = pickle.load(f,encoding='bytes')
	
	dataset[b'data'] = np.hstack((dataset1[b'data'],dataset2[b'data']))
	dataset[b'fine_labels'] = dataset1[b'fine_labels']
	
	if normalize:
		dataset[b'data'] = dataset[b'data'].astype(type)
		dataset[b'data'] /= 255.0
		testset[b'data'] = testset[b'data'].astype(type)
		testset[b'data'] /= 255.0

	if one_hot_label:
		dataset[b'fine_labels'] = _change_one_hot_label(dataset[b'fine_labels'],100)
		testset[b'fine_labels'] = _change_one_hot_label(testset[b'fine_labels'],100)

	if not flatten:
		dataset[b'data'] = dataset[b'data'].reshape(-1, 3, 32, 32)
		testset[b'data'] = testset[b'data'].reshape(-1, 3, 32, 32)
	
	if smooth:
		dataset[b'fine_labels'] = label_smoothing(dataset[b'fine_labels'],0.1)
		testset[b'fine_labels'] = label_smoothing(testset[b'fine_labels'],0.1)
	
	return (dataset[b'data'], dataset[b'fine_labels']), (testset[b'data'], testset[b'fine_labels'])

if __name__ == '__main__':
	(a,b),(c,d) = load_cifar100(False,False,True)
	print(a.shape)
	print(b.shape)