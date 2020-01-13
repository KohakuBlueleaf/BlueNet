import matplotlib.pyplot as plt
import numpy as np
from functions import *
from layer import *

ReLU = Relu()
ELU = Elu()
LReLU = Leaky()
x = np.arange(-5,5,0.001)

#plt.plot(x, arctan(x),label='arctan')
#plt.plot(x, arctan_grad(x),label='grad')
#plt.plot(x, tanh(x),label='tanh')
#plt.plot(x, sinh(x),label='sinh')
#plt.plot(x, cosh(x),label='cosh')
#plt.plot(x, sigmoid(x), label='sigmoid')
#plt.plot(x, softplus(x), label='softplus')
#plt.plot(x, ReLU.forward(x), label='ReLU')
#plt.plot(x, ELU.forward(x), label='ELU')
#plt.plot(x, LReLU.forward(x), label='LReLU')
plt.plot(x, gelu_erf(x), label='GELU')
plt.plot(x, gelu_erf_grad(x), label='GELU_grad')


plt.xlabel("Input")
plt.ylabel("Ouput")
plt.ylim(-1.2,4)
plt.axhline(0, color='black')
plt.axvline(0, color='black')
#plt.axhline(1, linestyle='-.', color='red')
#plt.axhline(-1, linestyle='-.', color='red')
plt.legend(loc='lower right')
plt.subplots_adjust(left=0.05, bottom=0.06, right=0.98, top=0.99, wspace=0.20, hspace=0.20)
plt.show()