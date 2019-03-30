import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
import numpy as np
import dlc_practical_prologue as prologue
import matplotlib.pyplot as plt

def create_model():
    model = nn.Sequential(
          nn.Linear(14*14, 5),
          nn.ReLU(),
          nn.Linear(5, 10),
          nn.ReLU()
        )
    return model

def convert_y_to_vect(y):
    y_vect = np.zeros((len(y), 10))
    for i in range(len(y)):
        y_vect[i, y[i]] = 1
    return y_vect

def convert_output_to_binary(output):
    max_indexes = output.max(0)[1]
    output_binary = torch.zeros(output.size())
    for i in range(list(max_indexes.size())[0]):
        output_binary[i][max_indexes[i]] = 1
    return output_binary


def train_model(model, train_input, train_target):
    criterion = nn.CrossEntropyLoss()
    eta = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=eta)
    batch_size = 10
    nb_epochs = 25

    for e in range(nb_epochs):
        for b in range(0, train_input.size(0), batch_size):
            #we add the [0] after train_input

            input=(train_input[b:b + batch_size][:, 0]).view(-1,196)
            print('Input size')
            print(input.size())
            #output = model(train_input[b:b + batch_size][:,0])
            output = model(input)
            print('Output size')
            print(output.size())


            #Here we get transform the target from a 2 into a [0 1 0 0 0 0 0 0 0 0]
            train_target_vect = torch.tensor(convert_y_to_vect(train_target[b:b + batch_size]))
            #print('Target vect size')
            #print(train_target_vect)

            #the output is not a neat binary of 0 or 1s so we transform the output into this neat binary
            #by making max entry of the 10 entry vector into 1 and all others 0
            output_binary = convert_output_to_binary(output)
            print(output_binary)
            loss = criterion(output_binary, train_target_vect)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

if __name__=='__main__':
    nb=1000
    train_input, train_labels, train_classes, test_input, test_labels, test_classes = prologue.generate_pair_sets(nb)
    #print(train_input.size())

    #loss on classifying the classes
    #loss on the integers themselves

    #print(train_classes)
    #print(train_labels)
    #1 entry is two images 14x14 matrices
    #print(train_input[1][1])
    plt.imshow(train_input[0][1], cmap='gray')
    plt.show()



    model = create_model()

    train_model(model, train_input, train_classes[:,0])