# coding: utf-8
import sys,os
sys.path.append("..")  
import pickle
import gzip
import numpy as np
from PIL import Image  
from bluenet.functions import _change_one_hot_label,label_smoothing


img_size = 784
dataset = {}
testset = {}

dataset_dir = os.path.dirname(os.path.abspath(__file__))
file = {
  'train_img':dataset_dir+'/data/emnist_data/emnist-train-images.gz',
  'train_label':dataset_dir+'/data/emnist_data/emnist-train-labels.gz',
  'test_img':dataset_dir+'/data/emnist_data/emnist-test-images.gz',
  'test_label':dataset_dir+'/data/emnist_data/emnist-test-labels.gz'
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

def load_emnist(normalize=True, flatten=True, one_hot_label=True, smooth=False, choose = 0, type=np.float32):
  dataset['data'] = load_imgs(file['train_img'])
  testset['data'] = load_imgs(file['test_img'])
  dataset['labels'] = load_labels(file['train_label'])
  testset['labels'] = load_labels(file['test_label'])

  dataset['data'] = dataset['data'].astype(type)
  testset['data'] = testset['data'].astype(type)

  if normalize:
    dataset['data'] /= 255
    testset['data'] /= 255

  if one_hot_label:
    dataset['labels'] = _change_one_hot_label(dataset['labels']-1,26)
    testset['labels'] = _change_one_hot_label(testset['labels']-1,26)

  if not flatten:
    dataset['data'] = dataset['data'].reshape(-1, 1, 28, 28).transpose(0,1,3,2)
    testset['data'] = testset['data'].reshape(-1, 1, 28, 28).transpose(0,1,3,2)
  
  if smooth:
    dataset['labels'] = label_smoothing(dataset['labels'],0.1)
    testset['labels'] = label_smoothing(testset['labels'],0.1)
  
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
  
  
  return (dataset['data'][data_choose], dataset['labels'][data_choose]), (testset['data'][test_choose], testset['labels'][test_choose])


if __name__ == '__main__':
  (x_train, t_train), (x_test, t_test) = load_emnist(normalize=False,flatten=False, one_hot_label=False)
  Image.fromarray(x_test[15001][0]).show()
  print(t_test[15001])
