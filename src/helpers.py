from torch import empty, Tensor
import torch
import math
from time import time


def generate_disc_set(nb):
    data_input = empty(size=(nb, 2)).uniform_(-1, 1)
    target = Tensor([[1, 0] if math.sqrt(x[0]**2 + x[1]**2) <= math.sqrt(2/math.pi) else [0, 1] for x in data_input])
    return data_input, target


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


def train_eval(n_epochs, model, criterion, optimizer, pytorch, x_train, y_train, x_test, y_test, batch_size):
    start = time()
    model.train()
    if pytorch:
        torch.set_grad_enabled(True)
        loss_vector = train_model_pytorch(n_epochs, x_train, model, criterion, y_train, optimizer, batch_size)
    else:
        torch.set_grad_enabled(True)
        loss_vector = train_model_diy(n_epochs, x_train, model, criterion, y_train, optimizer, batch_size)
    end = time()
    model.eval()
    error, prediction = compute_nb_errors(x_test, y_test, model, batch_size)
    return loss_vector, error*100, end - start, prediction
