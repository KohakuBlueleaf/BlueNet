import os
from os import listdir
from setuptools import setup
from Cython.Build import cythonize

datafiles = []
for path, dirs, files in os.walk('bluenet/dataset/data'):
	datafiles.append((path, [path+'/'+i for i in files]))


setup(
	name = 'bluenet',
	packages = ['bluenet','bluenet.dataset'],
  ext_modules = cythonize(
    ["bluenet/network.pyx","bluenet/layer.pyx","bluenet/activation.pyx","bluenet/optimizer.pyx","bluenet/functions.pyx"],
    compiler_directives={'language_level' : "3"}
  ),
	data_files = datafiles,
	version = '1.1',
	description = 'A neural network package based on numpy',
	author = 'BlueLeaf',
	author_email = 'apolloyeh0123@gmail.com',
	keywords = ['Neural Network'],
  zip_safe=False
)
