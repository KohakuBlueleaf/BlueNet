# coding: utf-8
import sys
sys.path.append("..")
import pickle
from random import randint as rand
from BlueNet.functions import _change_one_hot_label,label_smoothing
import numpy as np
from PIL import Image

dataset ={}
dataset1={}
dataset2={}

def load_cifar100(normalize=True, flatten=True, one_hot_label=False, smooth=False, type=np.float32):
	with open('BlueNet/database/cifar100_data/train1', 'rb',) as f:
		dataset1 = pickle.load(f,encoding='bytes')
	with open('BlueNet/database/cifar100_data/train2', 'rb',) as f:
		dataset2 = pickle.load(f,encoding='bytes')
	with open('BlueNet/database/cifar100_data/test', 'rb') as f:
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
	
def save_cifar100():
	with open('BlueNet/database/cifar100_data/train', 'rb',) as f:
		dataset = pickle.load(f,encoding='bytes')
	for keys,val in dataset.items():
		try:
			dataset1[keys] = np.vsplit(val,2)[0]
			dataset2[keys] = np.vsplit(val,2)[1]
		except ValueError:
			dataset1[keys] = val
			dataset2[keys] = val
			
	with open('BlueNet/database/cifar100_data/train1', 'wb',) as f:
		pickle.dump(dataset1,f)
	with open('BlueNet/database/cifar100_data/train2', 'wb',) as f:
		pickle.dump(dataset2,f)

if __name__ == '__main__':
	(a,b),(c,d) = load_cifar100(False,False,True)
	print(a.shape)
	print(b.shape)