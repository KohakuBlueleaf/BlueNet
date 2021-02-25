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
		dataset1 = pickle.load(f)
	
	with open(dataset_dir+'/data/cifar100_data/cifar100_data_train2', 'rb',) as f:
		dataset2 = pickle.load(f)
	
	with open(dataset_dir+'/data/cifar100_data/cifar100_data_test', 'rb') as f:
		testset = pickle.load(f)
	
	dataset['imgs'] = np.hstack((dataset1['imgs'],dataset2['imgs']))
	dataset['labels'] = dataset1['labels']
	
	if normalize:
		dataset['imgs'] = dataset['imgs'].astype(type)
		dataset['imgs'] /= 255.0
		testset['imgs'] = testset['imgs'].astype(type)
		testset['imgs'] /= 255.0

	if one_hot_label:
		dataset['labels'] = _change_one_hot_label(dataset['labels'],100)
		testset['labels'] = _change_one_hot_label(testset['labels'],100)

	if not flatten:
		dataset['imgs'] = dataset['imgs'].reshape(-1, 3, 32, 32)
		testset['imgs'] = testset['imgs'].reshape(-1, 3, 32, 32)
	
	if smooth:
		dataset['labels'] = label_smoothing(dataset['labels'],0.1)
		testset['labels'] = label_smoothing(testset['labels'],0.1)
	
	return (dataset['imgs'], dataset['labels']), (testset['imgs'], testset['labels'])

if __name__ == '__main__':
	(a,b),(c,d) = load_cifar100(False,False,True)
	print(a.shape)
	print(b.shape)
	print(c.shape)
	print(d.shape)