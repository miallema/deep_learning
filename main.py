import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
import numpy as np
import dlc_practical_prologue as prologue
import matplotlib.pyplot as plt

def create_model():
    model = nn.Sequential(
          nn.Linear(14, 196),
          nn.ReLU(),
          nn.Linear(196, 10),
          nn.ReLU()
        )
    return model


def train_model(model, train_input, train_target):
    criterion = nn.CrossEntropyLoss()
    eta = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=eta)
    batch_size = 1
    nb_epochs = 25

    for e in range(nb_epochs):
        for b in range(0, train_input.size(0), batch_size):
            #we add the [0] after train_input
            output = model(train_input[b:b + batch_size][:,0])
            loss = criterion(output, train_target[b:b + batch_size])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

if __name__=='__main__':
    nb=1000
    train_input, train_labels, train_classes, test_input, test_labels, test_classes = prologue.generate_pair_sets(nb)
    print(train_input.size())

    #loss on classifying the classes
    #loss on the integers themselves

    print(train_classes)
    #print(train_labels)
    #1 entry is two images 14x14 matrices
    #print(train_input[1][1])
    plt.imshow(train_input[0][1], cmap='gray')
    plt.show()



    model = create_model()

    train_model(model, train_input, train_classes[:,0])