# BlueNet
## A neural network module based on numpy

### =====================================

### Module I used
#### Numpy
#### Cupy
#### Scipy
#### Cupyx.scipy
#### pickle
#### gzip
### =====================================

## File
### 1.`Network.py`
#### Integrate most all the function that a network need(You can ignore it if you are a pro)

### 2.`layer.py`
#### Layer for neural network
#### This module has : 
#### Dense/Conv/DeConv/Max-Pool/Avg-Pool/Faltten/Batch Normalization/Dropout
#### ResLayer(block for ResNet)/ResLayerV2(Block for ResNetV2)/SoftmaxWithLoss
#### Embedding(Word Embedding)/TimeDense/LSTM/TimeLSTM/GRU/TimeGRU
#### sample:
```python
import numpy as np
from Activation import Sigmoid

#Dense
data = np.random.randn(10,100)
Layer = Dense(10,AF=Sigmoid)
result = Layer.forward(data)
print(result.shape)#(10,10)

#Conv
data = np.random.randn(10,1,25,25)
Layer = Conv({'f_num':16, 'f_size':5, 'pad':0, 'stride':2},AF=Sigmoid)
result = Layer.forward(data)
print(result.shape)#(10,16,11,11)
```

### 3.`Activation.py`
#### Activation Functions
#### This module has : 
#### Sigmoid/Relu/Leaky(LReLU)/Elu/GELU/ISRLU/ISRU
#### Softplus/Softsign/Softclip/SQNL/Tanh/Arctan

### 4.`optimizer.py` 
#### optimizer for updating the weights

### 5.`functions.py` 
#### Just functions

### 6.`config.py`
#### Have some usual model(AlexNet,VGG,ResNet)

### 7.`setting.py`
#### Choose GPU or CPU to compute
```python
#for CPU
import numpy as np
from scipy.special import erf
from numpy import exp

np = np
erf = erf
exp = exp

#for GPU
import cupy as np
import cupyx.scipy.special.erf as erf
from cupy import exp

np = np
erf = erf
exp = exp
```

### 8.`database`
#### Loader for database(Mnist/Emnits-letters/cifar-10)

