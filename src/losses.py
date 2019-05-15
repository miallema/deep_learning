class LossMSE:
    def __call__(self, prediction, targets):
        return self.apply(prediction, targets)

    def apply(self, prediction, targets):
        return self.loss(prediction, targets), self.dloss(prediction, targets)

    @staticmethod
    def loss(prediction, targets):
        return (targets - prediction).pow(2).mean()

    @staticmethod
    def dloss(prediction, targets):
        return -2 * (targets - prediction)

