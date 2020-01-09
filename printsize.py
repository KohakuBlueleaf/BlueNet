# coding: utf-8
from config import Network as model
from Network import Net

net = Net(model,data_shape=(1,28,28))

net.print_size()
