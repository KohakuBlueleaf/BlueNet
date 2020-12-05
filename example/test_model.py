## Import BlueNet
from blueNet.network import Net
from bluenet.layer import *

model = [
    MobileConv({'f_num':16, 'f_size':3, 'pad':1, 'stride':1}),
    MobileConv({'f_num':16, 'f_size':3, 'pad':1, 'stride':1}),
    MobileConv({'f_num':16, 'f_size':3, 'pad':1, 'stride':1}),
    Pool(2,2,2),
    
    MobileConv({'f_num':32, 'f_size':3, 'pad':1, 'stride':1}),
    MobileConv({'f_num':32, 'f_size':3, 'pad':1, 'stride':1}),
    MobileConv({'f_num':32, 'f_size':3, 'pad':1, 'stride':1}),
    Pool(2,2,2),
    
    MobileConv({'f_num':64, 'f_size':3, 'pad':1, 'stride':1}),
    MobileConv({'f_num':64, 'f_size':3, 'pad':1, 'stride':1}),
    MobileConv({'f_num':64, 'f_size':3, 'pad':1, 'stride':1}),
    Pool(2,2,2),
    
    Flatten(),
    Dense(output_size=512),
    Dense(output_size=10),
      
    SoftmaxWithLoss()
    ]

net = Net(model, (3,32,32))
net.print_size()