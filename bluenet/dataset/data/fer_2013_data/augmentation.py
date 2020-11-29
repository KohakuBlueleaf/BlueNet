from numpy import *
from pickle import *
from PIL import Image
from random import randint as ri
from random import shuffle
from scipy.ndimage.interpolation import rotate

def aug(img):
	new = []
	new.append(img)
	new.append(array([i[2:-3] for i in rotate(img,angle=6)[2:-3]]))
	new.append(array([i[2:-3] for i in rotate(img,angle=-6)[2:-3]]))
	new.append(array([i[7:-7] for i in rotate(img,angle=20)[7:-7]]))
	new.append(array([i[7:-7] for i in rotate(img,angle=-20)[7:-7]]))

	for i in range(len(new)):
		new.append(flip(new[i],axis=1))

	return [i.reshape(-1) for i in new]


with open('./fer2013_train.pydict','rb') as f:
	data = load(f)

imgs = data['imgs'].reshape(-1,48,48)
labels = data['labels']

num = imgs.shape[0]
new_datas = []

i = 1
for img, label in zip(imgs,labels):
	print(f'{i}/{num}',end='\r')
	new = aug(img)
	for new_img in new:
		new_datas.append([new_img,label])
	i+=1

shuffle(new_datas)
imgs = array([i[0] for i in new_datas])
labels = array([i[1] for i in new_datas])

data['imgs'] = imgs
data['labels'] = labels

with open('./fer2013_train_aug.pydict','wb') as f:
	dump(data,f)