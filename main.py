import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
import numpy as np
import dlc_practical_prologue as prologue
import matplotlib.pyplot as plt

def create_model():
    model = nn.Sequential(
          nn.Linear(14*14, 200),
          nn.ReLU(),
          nn.Linear(200, 10),
          nn.ReLU()
        )
    return model


def compute_nb_errors(model, input, target, mini_batch_size):
    errors = 0

    # we have too much data to give everything and giving one by one is too slow
    # so we give by minibatch
    for b in range(0, input.size(0), mini_batch_size):

        input_reformatted = (input[b:b + mini_batch_size][:, 0]).view(-1, 196)
        output = model(input_reformatted)
        #print(output)
        winner_output = torch.argmax(output, dim=1)
        #print(winner_output)

        target_vect=convert_y_to_vect(target.narrow(0, b, mini_batch_size))
        #print(target_vect)
        winner_target = torch.argmax(target_vect, dim=1)
        #print(winner_target)

        # argmax index le plus grand dans le vecteur output
        # argmax output et argmax target meme alors juste

        # we want the largest index to be the same in both
        error = winner_output - winner_target

        correct = error[error == 0]

        num_samples = output.size()
        num_correct = correct.size()

        nb_errors = num_samples[0] - num_correct[0]

        # print(nb_errors)
        errors += nb_errors

    return errors

def convert_y_to_vect(y):
    # transforms a target like a '2' into a [0 1 0 0 0 0 0 0 0 0] vector
    y_vect = np.zeros((len(y), 10))
    for i in range(len(y)):
        y_vect[i, y[i]] = 1
    return torch.tensor(y_vect).float()

def convert_output_to_binary(output):
    # transforms a vector like [0 0 0.2 0.5 0 0] into [0 0 0 1 0 0]
    max_indexes = output.max(0)[1]
    output_binary = torch.zeros(output.size())
    for i in range(list(max_indexes.size())[0]):
        output_binary[i][max_indexes[i]] = 1
    return output_binary


def train_model(model, train_input, train_target):
    criterion = nn.CrossEntropyLoss()
    eta = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=eta)
    batch_size = 100
    nb_epochs = 25

    for e in range(nb_epochs):
        for b in range(0, train_input.size(0), batch_size):
            #we add the [0] after train_input

            input_model=(train_input[b:b + batch_size][:, 0]).view(-1,196)

            #print('Input size')
            #print(input.size())

            output_model = model(input_model)
            #print('Output size')
            #print(output)
            #print(train_target[b:b + batch_size])


            # If we want MSEloss we can compute the train target binary vector
            # and compare it to the output as a neat binary vector
            # train_target_vect = convert_y_to_vect(train_target[b:b + batch_size])
            # the output is not a neat binary of 0 or 1s so we transform the output into this neat binary
            # by making max entry of the 10 entry vector into 1 and all others 0
            # output_binary = torch.tensor(convert_output_to_binary(output))

            loss = criterion(output_model, train_target[b:b + batch_size])
            #print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

if __name__=='__main__':
    mini_batch_size = 100

    #create samples
    nb=1000
    train_input, train_labels, train_classes, test_input, test_labels, test_classes = prologue.generate_pair_sets(nb)
    #print(train_input.size())

    #Notes
    #loss on classifying the classes
    #loss on the integers themselves

    #Data Exploration
    #print(train_classes)
    #print(train_labels)
    #1 entry is two images 14x14 matrices
    #print(train_input[1][1])
    plt.imshow(train_input[0][1], cmap='gray')
    plt.show()

    #standardize
    #mu, std = train_input.mean(0), train_input.std(0)
    #train_input.sub_(mu).div_(std)
    #test_input.sub_(mu).div_(std)

    #Create model
    model = create_model()
    #Train model
    train_model(model, train_input, train_classes[:,0])

    #Errors
    train_errors = compute_nb_errors(model, train_input, train_classes[:,0], mini_batch_size)
    print('Train errors:')
    print(train_errors)
    print(train_input.size(0))
    test_errors = compute_nb_errors(model, test_input, test_classes[:, 0], mini_batch_size)
    print('Test errors:')
    print(test_errors)
    print(train_input.size(0))


    print('Done')