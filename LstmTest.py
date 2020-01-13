from layer import *
import numpy as np
LSTM = TimeLSTM()
rn = np.random.randn
D, H=1000,500
data = rn(10,5,D)
LSTM.params['Wx'] = (rn(D, 4*H)/ np.sqrt(D)).astype('f')
LSTM.params['Wh'] = (rn(H, 4*H)/ np.sqrt(D)).astype('f')
LSTM.params['b'] = np.zeros(4*H).astype('f')
print('LSTM shape IN',LSTM.forward(data).shape)
loss = rn(10,5,H)
print('LSTM shape Out',(LSTM.backward(loss).shape))
for keys in (LSTM.grad):
	print('grad_'+keys,LSTM.grad[keys].shape)

print('')

GRU = TimeGRU()
rn = np.random.randn
D, H=1000,500
data = rn(10,5,D)
GRU.params['Wx'] = (rn(D, 3*H)/ np.sqrt(D)).astype('f')
GRU.params['Wh'] = (rn(H, 3*H)/ np.sqrt(D)).astype('f')
GRU.params['b'] = np.zeros(3*H).astype('f')
print('GRU shape IN',GRU.forward(data).shape)
loss = rn(10,5,H)
print('GRU shape Out',GRU.backward(loss).shape)
for keys in (GRU.grad):
	print('grad_'+keys,GRU.grad[keys].shape)