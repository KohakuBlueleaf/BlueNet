# BlueNet
## A neural network module based on numpy

## How to use?
### I haven't writen a `setup.py` for my module so you need to check all the file(include your code) is under the same path
### =====================================

## Module I used
### Numpy
### Cupy
### Scipy
### Cupyx.scipy
### pickle
### gzip
### =====================================

## File
### 1.`Network.py`
#### Integrate most all the function that a network need(You can ignore it if you are pro)

### 2.`layer.py`
#### Layer for neural network
#### We have : 
#### Dense/Conv/DeConv/Max-Pool/Avg-Pool/Faltten/Batch Normalization/Dropout
#### ResLayer(block for ResNet)/ResLayerV2(Block for ResNetV2)/SoftmaxWithLoss
#### Embedding(Word Embedding)/TimeDense/LSTM/TimeLSTM/GRU/TimeGRU

### 3.`Activation.py`
#### Activation Functions
#### We have : 
#### Sigmoid/Relu/Leaky(LReLU)/Elu/GELU/ISRLU/ISRU
#### Softplus/Softsign/Softclip/SQNL/Tanh/Arctan

### 4.`optimizer.py` 
#### optimizer for updating the weights

### 5.`functions.py` 
#### Functions for actavition functions or convolution or NLP

### 6.`config.py`
#### Have some usual model(VGG ResNet)

### 7. database
#### Loader for database(Mnist/Emnits-letters/cifar-10)

