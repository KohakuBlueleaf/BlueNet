# coding: utf-8
from config import Network as model
from Network import Net
net = Net(model,data_shape=(3,32,32))
net.print_size()
input()
'''
from GANconfig import Gen as G
from GANconfig import Dis as D
net = Net(G,data_shape=(1,9,9))
net.print_size()
net = Net(D,data_shape=(1,28,28))
net.print_size()
'''