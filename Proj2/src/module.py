from torch import empty
from abc import ABC, abstractmethod
from math import sqrt
from copy import deepcopy


'''
Abstract base class Module that inherits from ABC, contains abstractmethods forward
and backward, staticmethod initialize. @property and @training.setter are used to implement
the getter and setter.
'''
class Module(ABC):
    __train = None
    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def forward(self, input):
        pass

    @abstractmethod
    def backward(self, gradient):
        pass

    @property
    def training(cls):
        return cls.__train

    @training.setter
    def training(cls, new_training):
        cls.__train = new_training

    @staticmethod
    def initialize(n_out, n_in):
        std = sqrt(2 / (n_in + n_out))
        return empty([n_out, n_in]).normal_(0, std)


'''
Sequential inherits from Module and gets initialized with a list of modules. From 
the sequential module we change the class variable train from True to False and vise 
versa. Again using @property, we can get the modules. For the forward and backward pass
we simply iterate over the modules calling the desired function each time.
'''
class Sequential(Module):
    def __init__(self, *modules):
        self.__modules = []
        for module in modules:
            self.__modules.append(module)

    def train(self):
        Module.training = True

    def eval(cls):
        Module.training = False

    @property
    def modules(self):
        return self.__modules

    def forward(self, input):
        for module in self.modules:
            input = module.forward(input)
        return input

    def backward(self, gradient):
        for module in self.modules[::-1]:
            gradient = module.backward(gradient)
        return gradient


'''
The Linear module inherits from Module and implements the fully connected layer.
It has 5 class attributes. X, the input from the forward pass, w, the 2-d tensor of
weight parameters initialized using the Xavier initialization, dldw, the gradient of 
the weights, with the same dimensions as the weights, but initialized at zero. Further b,
the bias term initialized at 0.01 and dldb, the gradient of the bias initialized at zero.
'''
class Linear(Module):
    def __init__(self, n_in, n_out):
        self.__x = None
        self.__w = Module.initialize(n_in, n_out)
        self.__dldw = empty(n_in, n_out).zero_()
        self.__b = empty(1, n_out).fill_(0.01)
        self.__dldb = empty(1, n_out).zero_()

    #First the input gets copied and saved.
    #Then we return the linear transformation XW+b.
    def forward(self, input):
        self.__x = deepcopy(input)
        return self.__x.mm(self.__w) + self.__b

    #The gradient of the bias of a minibatch is the sum of 
    #the backpropagated gradient. The gradient of the weights is
    #the previous input transposed times the backpropagated gradient.
    #Finally we backpropagate the gradient times the current weights transposed.
    def backward(self, gradient):
        self.__dldb = gradient.sum(0).view(1, -1)
        self.__dldw = self.__x.t().mm(gradient)
        return gradient.mm(self.__w.t())
    
    #Getters and setters for attribute access in the optimizer step.
    
    @property
    def w(self):
        return self.__w

    @w.setter
    def w(self, new_w):
        self.__w = new_w

    @property
    def dldw(self):
        return self.__dldw

    @property
    def b(self):
        return self.__b

    @b.setter
    def b(self, new_b):
        self.__b = new_b

    @property
    def dldb(self):
        return self.__dldb

'''Dropout module inherits from Module. This module allows to set to zero
with probability p the input propagated forward, if the Module class variable
train is set to True, else the input is simply passed on.
'''
class Dropout(Module):
    def __init__(self, p):
        self.__p = p
        self.__mask = None

    #The randomly chosen mask needs to be stored for the backward propagatino.
    def forward(self, input):
        if Module.training:
            self.__mask = empty(input.shape).bernoulli_(1 - self.__p)
            return input * self.__mask / (1 - self.__p)
        else:
            return input

    #Only the gradient of the unmasked parameters gets propagated backwards.
    def backward(self, gradient):
        if Module.training:
            return gradient * self.__mask / (1 - self.__p)
        else:
            return gradient

'''Nonlinear activation ReLU inherits from abstract class Module. In the forward
pass the input gets saved before forwarding the input, whilesetting to zero all values, 
which are smaller than 0.
''' 
class ReLU(Module):
    def __init__(self):
        self.__x = None

    def forward(self, input):
        self.__x = deepcopy(input)
        return self.__x.apply_(lambda i: 0 if i < 0 else i)
    
    #The gradient of ReLU is 0 for x<=0 and 1 else. We backprop this multiplied with
    #the received gradient.
    def backward(self, gradient):
        return (self.__x > 0).float() * gradient

    
'''
Nonlinear activation Tanh inherits from abstract class Module. In the forward
pass the input gets saved before forwarding the input applied to the tanh. In the
backwards pass, backpropagated gradient gets multiplied with the derivative of Tanh applied
to the saved input from the forward pass.
'''
class Tanh(Module):
    def __init__(self):
        self.__x = None

    def forward(self, input):
        self.__x = deepcopy(input)
        return self.__x.tanh()

    def backward(self, gradient):
        return (1 - self.__x.tanh().pow(2)) * gradient
