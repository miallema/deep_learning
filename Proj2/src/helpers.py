from torch import empty, Tensor
import torch
import math
from time import time


#Function basically copied from practical session. We use the torch object Tensor here, but only here and not in
#the mini-framework !
def generate_disc_set(nb):
    data_input = empty(size=(nb, 2)).uniform_(-1, 1)
    target = Tensor([[1, 0] if math.sqrt(x[0]**2 + x[1]**2) <= math.sqrt(2/math.pi) else [0, 1] for x in data_input])
    return data_input, target


#Train method very heavily inspired by train methods written in the practical session. We tried to implement the 
#mini-framework such that the train method for the diy version and the PyTorch version would be maximally similar.
def train_model_diy(n_epochs, x_train, model, criterion, y_train, optimizer, batch_size):
    loss_vector = empty(n_epochs)
    for e in range(n_epochs):
        loss_epoch = 0
        for b in range(0, x_train.size(0), batch_size):
            output = model.forward(x_train.narrow(0, b, batch_size))
            loss, gradient = criterion(output, y_train.narrow(0, b, batch_size))
            loss_epoch += loss
            model.backward(gradient)
            optimizer.step()
        loss_vector[e] = loss_epoch
    return loss_vector


#Train method for PyTorch, basically copied from practical session.
def train_model_pytorch(n_epochs, x_train, model, criterion, y_train, optimizer, batch_size):
    loss_vector = empty(n_epochs)
    for e in range(n_epochs):
        loss_epoch = 0
        for b in range(0, x_train.size(0), batch_size):
            output = model.forward(x_train.narrow(0, b, batch_size))
            loss = criterion(output, y_train.narrow(0, b, batch_size))
            loss_epoch += loss.item()
            model.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_vector[e] = loss_epoch
    return loss_vector


#Method to compute the number of errors, again basically copied from practical session. 
def compute_nb_errors(data_input, data_target, model, batch_size):
    n_misclassified = 0
    num_samples = data_input.size(0)
    prediction = empty(num_samples).zero_()
    for b in range(0, num_samples, batch_size):
        batch_output = model.forward(data_input.narrow(0, b, batch_size))
        batch_target = data_target.narrow(0, b, batch_size)
        output_class = batch_output.max(1)[1]
        target_class = batch_target.max(1)[1]
        prediction[b:b+batch_size] = output_class
        n_misclassified += (output_class != target_class).sum()
    error = int(n_misclassified) / num_samples
    return error, prediction


#In order to keep the test.py file clean, we further implemented this train_eval function, which measures
#the time taken for training, then calculates the number of test errors and returns the loss, the error,
#the time and the predictions.
def train_eval(n_epochs, model, criterion, optimizer, pytorch, x_train, y_train, x_test, y_test, batch_size):
    start = time()
    model.train()
    if pytorch:
        torch.set_grad_enabled(True)
        loss_vector = train_model_pytorch(n_epochs, x_train, model, criterion, y_train, optimizer, batch_size)
    else:
        torch.set_grad_enabled(False)
        loss_vector = train_model_diy(n_epochs, x_train, model, criterion, y_train, optimizer, batch_size)
    end = time()
    model.eval()
    error, prediction = compute_nb_errors(x_test, y_test, model, batch_size)
    return loss_vector, error*100, end - start, prediction
