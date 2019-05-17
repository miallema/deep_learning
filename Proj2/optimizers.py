from module import Linear


'''
SGD is the class that implements the optimizer used in this mini-framework. 
It contains a model and a learning rate as object attribute. When the step
function is called it iterates over the modules of the model. In this mini-
framework only the linear layer contains parameters that need to be updated.
If the module is a linear layer, the weights and bias of the module get updated
by substracting the multiplication of the learning rate by the current values of 
the gradients. Then the gradients get set to zero. 

If the mini-framework were to be developed further, adding more layers, which contain
parameters, then the structure would have to be changed, implementing as suggested
an abstract method params() in the abstract base class Module, in order to get the 
parameters. For the scope of this project we found it more instructive to simply use
the built in Python isinstance() function.
'''
class SGD:
    def __init__(self, model, learning_rate=0.001):
        self.__model = model
        self.__learning_rate = learning_rate

    def step(self):
        for module in self.__model.modules:
            if isinstance(module, Linear):
                module.w -= self.__learning_rate * module.dldw
                module.b -= self.__learning_rate * module.dldb
                module.dldw.zero_()
                module.dldb.zero_()

