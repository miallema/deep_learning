'''
Class LossMSE implements the mean squared error, which is used in this mini-framework
as loss function, we implement it as a callable object, which returns the loss and 
the derivative of the loss given predictions and targets.
'''
class LossMSE:
    def __call__(self, prediction, targets):
        return self.apply(prediction, targets)

    def apply(self, prediction, targets):
        return self.loss(prediction, targets), self.dloss(prediction, targets)

    #The sum of the component-wise squared difference, divided by the total 
    #number of components in the tensor.
    @staticmethod
    def loss(prediction, targets):
        return (targets - prediction).pow(2).mean()

    #The derivative of LossMSE is -2 times the difference between the target and the prediction.
    @staticmethod
    def dloss(prediction, targets):
        return -2 * (targets - prediction)

