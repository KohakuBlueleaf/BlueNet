import numpy as np
import time

from BlueNet.Network import Net
from BlueNet.layer import Dense,SoftmaxWithLoss
from BlueNet.Activation import GELU
from BlueNet.optimizer import Adam
import BlueNet.dataset.mnist as mnist

(x_train,t_train),(x_test,t_test) = mnist.load_mnist(True, True, True, False, np.float32)
## load train set and test set                    Normalize Flat One-hot Smooth type

model = [Dense(550), Dense(10), SoftmaxWithLoss()]
net = Net(model, (1,28,28), GELU, Adam, 0.001, 0, 'xaiver', np.float32)
net.update()

batch_size = 300
train_size = x_train.shape[0]
iter_per_epoch = max((train_size // batch_size), 1)
max_acc = net.accuracy(x_test, t_test, 100)

round = int(input('Epoch:'))
print('\n┌──────────────────────────┐  ')
for j in range(round):
	start = time.time()
	if j != 0:
		print("│ =========================│  ")
	
	for i in range(iter_per_epoch):
		batch_mask = np.random.choice(train_size, batch_size) 	#Random choose data
		
		x_batch = x_train[batch_mask]
		t_batch = t_train[batch_mask]
		
		loss = net.train(x_batch, t_batch) 						#Train&Caculate the loss of the net
		print('│ Epoch %2d  Loss:%5f  │  '%(j+1,loss),end='\r',flush=True)
	
	cost = time.time()-start
	test_acc = net.accuracy(x_test,t_test)
	train_acc = net.accuracy(x_train,t_train)
	
	if test_acc>max_acc:
		max_acc = test_acc 
		net.save() 												#Save the parameters
	
	print("│ Epoch {:3}  Test Acc:{:5.5}│  ".format(j+1,str(test_acc*100)))
	print("│           Train Acc:{:5.5}│  ".format(str(train_acc*100)))
	print("│           Cost Time:{:5.5}│  ".format(str(cost)))
print('└──────────────────────────┘  ')

net.update()
print("Final Accuracy: %2f%%"%(100*net.accuracy(x_test, t_test)))
