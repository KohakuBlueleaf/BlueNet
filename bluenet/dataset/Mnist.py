# coding: utf-8
import sys,os
sys.path.append("..") 
import os.path
import pickle
import numpy as np
from bluenet.functions import _change_one_hot_label,label_smoothing


dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/data/mnist_data/mnist.pkl"

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784

def load_mnist(normalize=True, flatten=True, one_hot_label=True, smooth=False, type=np.float32):
	with open(save_file, 'rb') as f:
		dataset = pickle.load(f)

	if normalize:
		for key in ('train_img', 'test_img'):
			dataset[key] = dataset[key].astype(type)
			dataset[key] /= 255.0

	if one_hot_label:
		dataset['train_label'] = _change_one_hot_label(dataset['train_label'],10)
		dataset['test_label'] = _change_one_hot_label(dataset['test_label'],10)

	if not flatten:
		for key in ('train_img', 'test_img'):
			dataset[key] = dataset[key].reshape(-1, 1, 28, 28)
	else:
		for key in ('train_img', 'test_img'):
			dataset[key] = dataset[key].reshape(-1, 784)
	
	if smooth:
		dataset['train_label'] = label_smoothing(dataset['train_label'],0.1)
		dataset['test_label'] = label_smoothing(dataset['test_label'],0.1)
	
	return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])
