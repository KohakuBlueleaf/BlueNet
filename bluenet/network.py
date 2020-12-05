# coding: utf-8

#Original
from bluenet.activation import *
from bluenet.optimizer import *
from bluenet.functions import *
from bluenet.layer import *
from bluenet.usualmodel import LeNet

#native(or usual)
from bluenet.setting import _np,device
from copy import copy,deepcopy
from time import time
import sys,os
import shutil
import json
import numpy
import pickle

sys.setrecursionlimit(100000)
rn = _np.random.randn
uni = _np.random.uniform
norm = _np.random.normal
D3 = {'Conv','DeConv','ResLayer','Flatten','Mobile'}
D2 = {'Dense'}
PRE = {'Conv','DeConv','Softmax','ResLayer','Mobile'}

class Net:
  def __init__(self, network = [], **kwargs):
    self.net = []
    for i in network:
      self.net.append(i)
    self.layers = len(network)                #amount of layers
    if kwargs:
      self.initialize(**kwargs)
  
  def initialize(self, shape=(1,28,28), af=Relu, opt = Adam, rate=0.001\
          , init_std=0.005, init_mode='normal', dtype=_np.float32, **kwargs):
    self.structure = {
      'network': '[{}]'.format(','.join((i.str for i in self.net))),
      'shape': shape,
      'af': af,
      'opt': opt,
      'rate': rate,
      'init_std': init_std,
      'init_mode': init_mode,
      'dtype': dtype
    }
    
    if 'optimizer' in kwargs:
      self.optimizer = kwargs['optimizer']
    else:
      self.optimizer = opt
    self.learing_rate = rate
    self.in_shape = shape
    self.dtype = dtype

    #initial process
    init = rn(*shape,dtype=dtype)  #data for initial
    j = 0
    for i in range(self.layers):
      name = self.net[i].name
      
      if j == 0:
        #set the shape of data. Falt the data if first layer is dense
        if name in D3:
          init = init.reshape(1,init.shape[0],init.shape[1],init.shape[2])
        elif name in D2:
          init = init.reshape(1,init.size)
      
      self.net[i].shape_in = init.shape[1:]        #save the input shape
      self.net[i].optimizer = self.optimizer(rate)        #set Optimizer
      if not self.net[i].af:
        self.net[i].af = af()
      self.net[i].dtype = dtype

      if init_mode!='normal':
        init_std_b=0
      else:
        init_std_b = init_std

      if len(init.shape)==2:
        init_w = get_initializer(init, init_std, init_mode, dtype)
      elif len(init.shape)==4:
        init_w = get_conv_initializer(init, init_std, init_mode, dtype)

      #Convolution
      if name == 'Conv' or name == 'DeConv':
        self.net[i].params['b'] *= init_std_b
        FN, C, S = self.net[i].f_num, init.shape[1], self.net[i].f_size

        #set initializer for Conv
        #Convolution
        if name == 'Conv':
          #weight's shape is (F_num, input_channel, F_size, F_Size)
          self.net[i].params['W'] = init_w(FN, C, S, S)
          out = self.net[i].forward(init)            #data to set next layer
          
          N, out_C, out_H, out_W = out.shape
          self.net[i].flops = ((C*S**2))*out_H*out_W*out_C  #caculate the FLOPs
          self.net[i].size = FN*C*S*S + FN          #caculate the amount of parameters
        
        #Transpose Convolution
        else:
          #weight's shape is (Input_channel,F_Num,F_size,F_Size)
          self.net[i].params['W'] = init_w(C, FN, S, S)
          out = self.net[i].forward(init)            #data to set next layer
          
          N, out_C, out_H, out_W = out.shape
          self.net[i].flops = ((C*S**2)-1)*out_H*out_W*out_C  #caculate the FLOPs
          self.net[i].size = self.net[i].params['W'].size    #caculate the amount of parameters
      
        init = out
          
      elif name == 'BatchNorm':
        self.net[i].size = 2
      
      #ResLayer(Block of ResNet)
      elif name == 'ResLayer':
        #set Activation Function
        self.net[i].af = af
        #see layer.py(In fact the function is same as here)                          
        init = self.net[i].initial(init,init_std,init_mode,af,opt,rate,dtype)
      
      #Fully connected layer
      elif name == 'Dense':
        out_size = self.net[i].output_size
        #weight's shape is (input_size,output_size)
        self.net[i].params['W'] = init_w(init.size, out_size)
        self.net[i].params['b'] *= init_std_b
        self.net[i].flops = init.shape[1]*out_size
        self.net[i].size = init.size*out_size + out_size
        
      #ResLayer(Block of ResNet)
      elif name == 'ResLayer':
        #set Activation Function
        self.net[i].af = af                              
        #see layer.py(In fact the function is same as here)
        init = self.net[i].initial(init,init_std,init_mode,af,opt,rate,dtype)
      
      elif name == 'Mobile':
        if not self.net[i].af:
          self.net[i].af = af
        init = self.net[i].initial(init,init_std,init_mode,af,opt,rate,dtype)
      
      elif name == 'TimeDense':
        T = init.shape[1]
        D = init.shape[2]
        H = self.net[i].output_size
        self.net[i].params['W'] = init_w(D, H)
        self.net[i].params['b'] = _np.ones(H)*init_std_b
        self.net[i].flops = T*D*H+H  
        self.net[i].size = D*(H+1)
        
      elif name == 'TimeLSTM':
        T = init.shape[1]
        D = init.shape[2]
        H = self.net[i].node
        self.net[i].params['Wx'] = _np.array([init_w(D, H) for i in range(4)]).reshape(D,H*4)
        self.net[i].params['Wh'] = _np.array([init_w(H, H) for i in range(4)]).reshape(H,H*4)
        self.net[i].params['b'] = _np.ones(4*H)*init_std_b
        self.net[i].flops = T*D*4*H+T*H*4*H  
        self.net[i].size = (D+H+1)*4*H
      
      elif name == 'TimeGRU':
        T = init.shape[1]
        D = init.shape[2]
        H = self.net[i].node
        self.net[i].params['Wx'] = _np.array([init_w(D, H) for i in range(3)]).reshape(D,H*3)
        self.net[i].params['Wh'] = _np.array([init_w(H, H) for i in range(3)]).reshape(H,H*3)
        self.net[i].params['b'] = _np.ones(3*H)*init_std_b
        self.net[i].flops = T*D*3*H+T*H*3*H
        self.net[i].size = (D+H+1)*3*H

      else:
        pass
      
      #these layers don't need to caculate the data for next layer so we just skip it
      if name not in PRE:
        try:
          init = self.net[i].forward(init)
        except:
          print(init.shape)
          print(self.net[i].params['W'].shape)
      
      #save the output shape
      self.net[i].shape_out = init.shape[1:]
      j += 1
    
    for i in range(self.layers):
      try:
        for x in self.net[i].params.keys():
          try:
            self.net[i].params[x] = self.net[i].params[x].asdtype(dtype)
          except AttributeError:
            pass
      except:
        pass
    
    self.save('./temp/')
    self.update('./temp/weight')
    shutil.rmtree('./temp/')
  
  def set_af(self,af=GELU):
    for layer in self.net:
      layer.af = af()
    
    self.structure['af'] = af

  def __copy__(self):
    new = Net()
    new.net = deepcopy(self.net)
    new.layers = self.layers
    
    return new

  #print the model(About layer/amount of parameter/FLOPS...)
  def print_size(self):
    total = 0    #Total amount of parameters
    total_f = 0   #Total FLOPs
    
    #print the table
    print("┌───────────┬───────┬──────────┬──────────────┬─────────────┐")
    print("│   Layer   │ MFLOPs│  Params  │   Shape(In)  │  Shape(Out) │")
    for i in self.net:
      try:
        total += i.size
        total_f += i.flops
        print("├───────────┼───────┼──────────┼──────────────┼─────────────┤")
        print(f"│{i.name:^11}│{str(i.flops/1000000)[:5]:^7}│{i.size:>10}│{str(i.shape_in).replace(' ',''):>14}│{str(i.shape_out).replace(' ',''):>13}│")
      except AttributeError:
        pass
        
    print("├───────────┼───────┼──────────┼──────────────┼─────────────┤")
    print("│   Total   │{:^7}│{:>10}│              │             │".format(str(total_f/1000000)[:5],total))
    print("└───────────┴───────┴──────────┴──────────────┴─────────────┘")  

  def test_gradient(self, batch=10, hist=False, hist_set=(-1.05,1.05,0.01), p=False):
    test_data = rn(batch, *self.in_shape, dtype=self.dtype)
    all_data = []
    
    for layer in self.net:
      if layer.name in {'Softmax','DropOut'}:
        continue

      test_data = layer.forward(test_data)
      if layer.name in {'Flatten','BFlatten','Max-Pool','Avg-Pool'}:
          continue
      
      all_data.append(test_data)
      var =  _np.var(test_data)

      if p:
        print('var:{} std:{}'.format(str(var)[:5],str(_np.sqrt(var))[:5]))
    if hist:
      return [_np.histogram(data, bins=_np.arange(*hist_set))[0] for data in all_data]
    else:
      return [data.get() for data in all_data]

  def LSUVlize(self, trials=1, levels=1, tol=0.15, batch=10):
    test_data = rn(1, *self.in_shape, dtype=self.dtype)
    shape = []
    for layer in self.net:
      shape.append(test_data.shape[1:])
      if layer.name in {'Softmax','DropOut'}:
        continue
      if layer.name in {'Flatten','BFlatten'}:
        test_data = layer.forward(test_data)
      else:
        test_data = layer.forward(test_data)

    for layer, shape in zip(self.net, shape):
      if layer.name in {'Softmax','DropOut','Flatten','BFlatten','Max-Pool','Avg-Pool','BatchNorm'}:
        continue
      
      #temp_af = type(layer.af)
      #layer.af = ID()

      for _ in range(trials):
        test_data = rn(batch, *shape, dtype=self.dtype)
        var = float(_np.var(layer.forward(test_data)+1e-7))
        if abs(var-1)<tol:
          break
        for key in layer.params:
          layer.params[key] *= (1/_np.sqrt(var))**(1/levels)
      
      #layer.af = temp_af()

  #forward process. DropOut is set OFF. SoftmaxWithLoss return the answer
  def process(self,input,drop=False):
    output = _np.asarray(input)
    
    for i in range(self.layers):
      if self.net[i].name == 'DropOut' and drop:
        output = self.net[i].forward(output)
      
      if self.net[i].name == 'Softmax':
        output = self.net[i].forward(output,0,False)
      else:
        output = self.net[i].forward(output)
    
    return output
    
  #forward process. DropOut is set ON. SoftmaxWithLoss return the loss
  def forward(self, input, t=None, loss_function=None):
    output = self.process(input,drop=True)
        
    if loss_function is not None and t is not None:
      t = _np.asarray(t)
      return output,loss_function(output,t)
    
    return output
  
  #Backpropagation (will save the gradients)
  def backward(self,error):
    self.net.reverse()
    
    #backpropagation in order
    for i in range(self.layers):
      error = self.net[i].backward(error)
    self.net.reverse()
    
    #return final error(for GAN's Discriminator or others)
    return error
  
  #Backward + train(Change the parameters)
  def back_train(self,error):
    error = self.backward(error)          #backpropagation first
    
    for i in range(self.layers):
      #call the train function
      self.net[i].train()
    
    return error
  
  #Train consist of forward, backtrain, call the optimizer
  def train(self, input, t, loss_function=cross_entropy_error):
    output = self.forward(input, t)      #forward
    t = _np.asarray(t)
    
    try:
      loss = loss_function(output,t)
    except:
      print(dtype(t), dtype(output))
    
    error = (output-t)/t.shape[0]
    self.back_train(error)
    
    return loss
  
  def pre_train_for_conv(self, all_data, batch_size=100):
    all_data = numpy.asarray(all_data[:10000])
    
    train_size = all_data.shape[0]
    iter_num = max(1, train_size//batch_size)
    if iter_num==1:
      batch_size = train_size
    
    layer_num=1
    print('\n┌────────────────────────────┐  ')
    print('│ Pre training start       │  ')
    for i in range(len(self.net)):
      layer = self.net[i]
      layer.optimizer = self.optimizer()
      start = time()
      
      if layer.name=='Conv' or layer.name=='Mobile':
        print("│ ===========================│  ")
        
        other_layers = self.net[:i]
        init = _np.asarray(all_data[:1])
        for other_layer in other_layers:
          init = other_layer.forward(init)
        
        if layer.f_size == 2*layer.pad+1:
          pre_train_layer = Conv({'f_num':init.shape[1], 'f_size':layer.f_size, 'stride':layer.stride, 'pad':layer.pad})
        else:
          size = layer.f_size-2*layer.pad
          pre_train_layer = DeConv({'f_num':init.shape[1], 'f_size':size, 'stride':layer.stride})
        
        pre_train_layer.af = dtype(layer.af)()
        pre_train_layer.optimizer = dtype(layer.optimizer)(0.0001)
        
        for _ in range(iter_num):
          data_mask = numpy.random.choice(train_size, batch_size)
          data = _np.asarray(all_data[data_mask])
          
          for other_layer in other_layers:
            data = other_layer.forward(data)
            
          #forward and backward
          forward = pre_train_layer.forward(layer.forward(data))
          error = (forward-data)/batch_size
          layer.backward(pre_train_layer.backward(error))
          
          #train
          layer.train()
          pre_train_layer.train()
          
          loss = RMS(data,forward)
          print('│ Layer {:<4}  Loss     :{:<5}│  '.format(i+1,str(loss)[:5]), end='\r')
        
        layer.optimizer = self.optimizer(self.learing_rate)
      else:
        layer_num += 1
        continue
      
      end = time()
      cost = end-start
      
      print("│ Layer {:<4}       Loss:{:<5}│  ".format(i+1,str(loss)[:5]))
      print("│             Cost Time:{:<5}│  ".format(str(cost)[:5]))
    
    print('└────────────────────────────┘  ')

  def reset(self):
    for i in range(self.layers):
      if self.net[i].name == 'TimeLSTM':
        self.net[i].reset_state()
  
  #caculate the accuracy of the net
  def accuracy(self, x, t, batch_size = 100, print_the_result = False):
    ac = 0                                #amount of correct answer
    for i in range(x.shape[0]//batch_size):                #process 10 datas in a time
      batch = numpy.arange(i*batch_size, batch_size+i*batch_size)    #choose the data in order
      
      x_batch = _np.asarray(x[batch])
      t_batch = _np.asarray(t[batch])
      
      y = self.process(x_batch)
      y = _np.argmax(y, axis=1)      
      tt = _np.argmax(t_batch, axis=1)
      ac += _np.sum(y == tt)                      #save the amount of correct answer
      
    accuracy = ac / x.shape[0]
    if print_the_result:
      print(ac,'/',x.shape[0],sep='')
    
    return accuracy
  
  #caculate the loss of the net(CEE)
  def loss(self, x, t): 
    t = _np.asarray(t)
    y = self.process(x)
    loss = cross_entropy_error(y, t)
    
    return loss
  
  #caculate the loss of the net(MSE)
  def loss_MSE(self, x, t): 
    t = _np.asarray(t)
    y = self.process(x)
    
    return mean_squared_error(y, t)  
  
  def loss_RMS(self, x, t):
    t = _np.asarray(t)
    y = self.process(x)
    
    return RMS(y,t)
  
  #Load the parameters
  def update(self,folder='./'):
    for i in range(self.layers):
      #call every layer's load function
      try:
        self.net[i].load(str(i+1),folder)
      except FileNotFoundError:
        pass
  
  #Load the parameters
  def load(self,folder='./'):
    try:
      with open(f'{folder}structure', 'rb') as f:
        self.structure = pickle.load(f)
      self.structure['network'] = eval(self.structure['network'])
      self.__init__(**self.structure)
      self.update(folder+'weight')
      return True
    except FileNotFoundError:
      pass
    except EOFError:
      pass
    return False

  #Save the parameters
  def save(self,folder='./'):
    if folder[-1] not in {'\\','/'}:
      folder += '/'

    if not os.path.isdir(folder):
      os.mkdir(folder)
    if not os.path.isdir(folder+'weight'):
      os.mkdir(folder+'weight')
    

    with open(f'{folder}structure', 'wb') as f:
      pickle.dump(self.structure,f)

    for i in range(self.layers):
      self.net[i].save(str(i+1), folder+'weight')