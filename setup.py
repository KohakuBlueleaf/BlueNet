import os
from os import listdir
from subprocess import run
from distutils.core import setup

package_path = 'Lib/site-packages/BlueNet/Dataset'
all_files = {}
data_files = []
queue = []

try:
	path = 'BlueNet/Dataset'
	Dataset = listdir(path)
	for i in Dataset:
		queue.append((path,i))

	while queue:
		path,now = queue.pop(0)
		this = '{}/{}'.format(path,now)
		if os.path.isdir(this):
			for i in listdir(this):
				queue.append((this,i))
		else:
			target_dir = package_path+path[15:]
			
			if target_dir not in all_files:
				all_files[target_dir]=[]
			all_files[target_dir].append(this)

	for i in all_files:
		data_files.append((i,all_files[i]))
except:
	pass

setup(
	name = 'BlueNet',
	packages = ['BlueNet'],
	data_files = data_files,
	version = '1.0',
	description = 'My first project',
	author = 'BlueLeaf',
	author_email = 'apolloyeh0123@gmail.com',
	keywords = ['Neural Network'],
)
