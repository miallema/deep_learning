from module import Linear

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

