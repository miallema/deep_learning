import torch 
from torch import Tensor
import dlc_practical_prologue as prologue
from time import time
import matplotlib.pyplot as plt

if __name__ == '__main__':
    tic = time()
    nb = 1000
    train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(nb)
    plt.imshow(train_input[0][1], cmap='gray')
    print(train_input[0:100 +100][:,0].size())
    print(train_input.size(0))




