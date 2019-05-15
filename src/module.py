from torch import empty
from abc import ABC, abstractmethod
from math import sqrt
from copy import deepcopy


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


class Linear(Module):
    def __init__(self, n_in, n_out):
        self.__x = None
        self.__w = Module.initialize(n_in, n_out)
        self.__dldw = empty(n_in, n_out).zero_()
        self.__b = empty(1, n_out).fill_(0.01)
        self.__dldb = empty(1, n_out).zero_()

    def forward(self, input):
        self.__x = deepcopy(input)
        return self.__x.mm(self.__w) + self.__b

    def backward(self, gradient):
        self.__dldb = gradient.sum(0).view(1, -1)
        self.__dldw = self.__x.t().mm(gradient)
        return gradient.mm(self.__w.t())

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


class Dropout(Module):
    def __init__(self, p):
        self.__p = p
        self.__mask = None

    def forward(self, input):
        if Module.training:
            self.__mask = empty(input.shape).bernoulli_(1 - self.__p)
            return input * self.__mask / (1 - self.__p)
        else:
            return input

    def backward(self, gradient):
        if Module.training:
            return gradient * self.__mask / (1 - self.__p)
        else:
            return gradient


class ReLU(Module):
    def __init__(self):
        self.__s = None

    def forward(self, input):
        self.__s = deepcopy(input)
        return self.__s.apply_(lambda i: 0 if i < 0 else i)

    def backward(self, gradient):
        return (self.__s > 0).float() * gradient


class Tanh(Module):
    def __init__(self):
        self.__s = None

    def forward(self, input):
        self.__s = deepcopy(input)
        return self.__s.tanh()

    def backward(self, gradient):
        return (1 - self.__s.tanh().pow(2)) * gradient
