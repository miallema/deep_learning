import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
import numpy as np
import dlc_practical_prologue as prologue
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2)
        self.fc1 = nn.Linear(4 * 64, 100)
        self.fc2 = nn.Linear(100, 24)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
        x = F.relu(self.fc1(x.view(-1, 4 * 64)))
        x = self.fc2(x)
        return x

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.fc1 = nn.Linear(4 * 64, 180)
        self.fc2 = nn.Linear(180, 24)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(-1, 4 * 64)))
        x = self.fc2(x)
        return x

class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=5)
        #self.drop1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(4 * 64, 180)
        self.fc2 = nn.Linear(180, 24)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        #x = self.drop1(x)
        x = F.relu(self.fc1(x.view(-1, 4 * 64)))
        x = self.fc2(x)
        return x


def compute_nb_errors(model, input, target_classes, target_labels, mini_batch_size):
    errors_num0 = 0
    errors_num1 = 0
    errors_comparison = 0

    # we have too much data to give everything and giving one by one is too slow
    # so we give by minibatch
    for b in range(0, input.size(0), mini_batch_size):

        input_model = input[b: b + mini_batch_size]
        output = model(input_model)
        #print(output.size())

        winner_output0 = torch.argmax(output[:, 2:12], dim=1)
        winner_output1 = torch.argmax(output[:, 12:22], dim=1)
        winner_comparison = torch.argmax(output[:, 0:2], dim=1)

        target_classes_vect0 = prologue.convert_to_one_hot_labels(torch.zeros(10), target_classes[b: b + mini_batch_size, 0])
        target_classes_vect1 = prologue.convert_to_one_hot_labels(torch.zeros(10), target_classes[b: b + mini_batch_size, 1])

        winner_target0=torch.argmax(target_classes_vect0, dim=1)
        winner_target1=torch.argmax(target_classes_vect1, dim=1)
        winner_target_comparison = target_labels[b: b + mini_batch_size]

        #print('Winner 0')
        #print(winner_output0.size())
        #print(winner_target0.size())
        #print(target_classes_vect0.size())
        #print('Winner 1')
        #print(winner_output1.size())
        #print(winner_target1.size())
        #print('Comparison winner')
        #print(winner_comparison.size())
        #print(winner_target_comparison.size())


        # we want the largest index to be the same in both
        error_num0 = winner_output0 - winner_target0
        error_num1 = winner_output1 - winner_target1
        error_comparison = winner_comparison -winner_target_comparison

        correct0 = error_num0[error_num0 == 0]
        correct1 = error_num1[error_num1 == 0]
        correct_comparison = error_comparison[error_comparison == 0]

        num_samples = output.size()
        num_correct0 = correct0.size()
        num_correct1 = correct1.size()
        num_correct_comparison = correct_comparison.size()

        nb_errors0 = num_samples[0] - num_correct0[0]
        nb_errors1 = num_samples[0] - num_correct1[0]
        nb_errors_comparison = num_samples[0] - num_correct_comparison[0]


        # print(nb_errors)
        errors_num0 += nb_errors0
        errors_num1 += nb_errors1
        errors_comparison += nb_errors_comparison


    return errors_num0, errors_num1, errors_comparison




def train_model(model, train_input, train_target, train_labels):
    criterion = nn.CrossEntropyLoss()
    eta = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=eta)
    batch_size = 100
    nb_epochs = 25

    for e in range(nb_epochs):
        #print(e)
        for b in range(0, train_input.size(0), batch_size):

            input_model = train_input[b: b + batch_size]
            #print('Input size')
            #print(input_model.size())

            #output_model outputs for example: [0,1,3,5]
            output_model = model(input_model)

            #print('Output obtained')
            #print(output_model.size())


            loss_comparison = criterion(output_model[:,0:2], train_labels[b:b + batch_size])
            #print('Loss 1 done')
            loss_number1 = criterion(output_model[:,2:12], train_target[b:b + batch_size][:,0])
            #print('Loss 2 done')
            loss_number2 = criterion(output_model[:,12:22], train_target[b:b + batch_size][:,1])
            #print('Loss 3 done')

            loss = 1/2*loss_comparison + 1/4*loss_number1 + 1/4*loss_number2
            #print(loss)
            optimizer.zero_grad()
            #print('optimizer zero grad done')
            loss.backward()
            #print('loss backward done')
            optimizer.step()
            #print('optimizer step done')

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
    model1 = Net()
    model2 = Net2()
    model3 = Net3()
    models = [model1, model2, model3]

    for model in models:
        print(model)
        #Train model
        train_model(model, train_input, train_classes, train_labels)

        #Calculate Train and Test error percentages
        errors_num0_train, errors_num1_train, errors_comparison_train = compute_nb_errors(model, train_input, \
                                                                        train_classes, train_labels, mini_batch_size)

        errors_num0_test, errors_num1_test, errors_comparison_test = compute_nb_errors(model, test_input, \
                                                                                          test_classes, test_labels,
                                                                                          mini_batch_size)

        print('Comparison Test error: {} %, First number Test error: {} %, Second number Test error: {} % ' \
                                    .format(errors_comparison_test*100/1000,errors_num0_test*100/1000,errors_num1_test*100/1000))
        print('Comparison Train error: {} %, First number Train error: {} %, Second number Train error: {} %' \
                  .format(errors_comparison_train*100/1000, errors_num0_train*100/1000, errors_num1_train*100/1000))

    print('Done')