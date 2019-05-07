import torch
from torch import nn
import dlc_practical_prologue as prologue

#global variables
mini_batch_size = 100
nb_epoch = 25


def error_calculator(target_classes_vect, winner_output, num_samples):
    winner_target=torch.argmax(target_classes_vect, dim=1)
    error_num = winner_output - winner_target
    correct = error_num[error_num == 0]
    num_correct = correct.size()
    nb_errors = num_samples[0] - num_correct[0]
    return nb_errors


def compute_nb_errors(model, input_model, target_classes, target_labels, mini_batch_size,
                      kernel1, kernel2, kernel3, kconv3):
    errors_num0 = 0
    errors_num1 = 0
    errors_comparison = 0

    # we have too much data to give everything and giving one by one is too slow
    # so we give by minibatch
    for b in range(0, input_model.size(0), mini_batch_size):
        # evaluate model on the minibatch
        input_model_minibatch = input_model[b: b + mini_batch_size]
        model.eval()
        output = model(input_model_minibatch, kernel1, kernel2, kernel3, kconv3)

        num_samples = output.size()
        # use argmax to determine which number is predicted and the result of the comparison
        winner_output0 = torch.argmax(output[:, 2:12], dim=1)
        winner_output1 = torch.argmax(output[:, 12:22], dim=1)
        winner_comparison = torch.argmax(output[:, 0:2], dim=1)

        # convert the numbers into a 10 sized binary vector
        target_classes_vect0 = prologue.convert_to_one_hot_labels(torch.zeros(10),
                                                                  target_classes[b: b + mini_batch_size, 0])
        target_classes_vect1 = prologue.convert_to_one_hot_labels(torch.zeros(10),
                                                                  target_classes[b: b + mini_batch_size, 1])

        # computer error for the prediction of each of the two numbers
        nb_errors0 = error_calculator(target_classes_vect0, winner_output0, num_samples)
        nb_errors1 = error_calculator(target_classes_vect1, winner_output1, num_samples)

        # compute error for the comparison of the two numbers
        winner_target_comparison = target_labels[b: b + mini_batch_size]
        error_comparison = winner_comparison - winner_target_comparison
        correct_comparison = error_comparison[error_comparison == 0]
        num_correct_comparison = correct_comparison.size()
        nb_errors_comparison = num_samples[0] - num_correct_comparison[0]

        # add errors of the minibatch to the errors tensor
        errors_num0 += nb_errors0
        errors_num1 += nb_errors1
        errors_comparison += nb_errors_comparison

    errors_comparison = errors_comparison * 100 / 1000
    errors_num0 = errors_num0 * 100 / 1000
    errors_num1 = errors_num1 * 100 / 1000

    return errors_num0, errors_num1, errors_comparison




def train_model(model, auxiliary, train_input, train_target, train_labels, nb_epochs, eta, \
                kernel1, kernel2, kernel3, kconv3):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=eta)
    batch_size = 100

    loss_per_epochs = torch.zeros([nb_epochs, 1])
    for e in range(nb_epochs):
        loss_per_batches= torch.zeros([1, len(range(0, train_input.size(0), batch_size))+1])
        batch = 0

        for b in range(0, train_input.size(0), batch_size):

            input_model = train_input[b: b + batch_size]

            model.train()
            output_model = model(input_model, kernel1, kernel2, kernel3, kconv3)

            loss_comparison = criterion(output_model[:, 0:2], train_labels[b:b + batch_size])
            loss_number1 = criterion(output_model[:, 2:12], train_target[b:b + batch_size][:, 0])
            loss_number2 = criterion(output_model[:, 12:22], train_target[b:b + batch_size][:, 1])

            if auxiliary == True:
                loss = 1 / 2 * loss_comparison + 1 / 4 * loss_number1 + 1 / 4 * loss_number2
            else:
                loss = loss_comparison

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch += 1

            comparison_error_batch, _, _ = compute_nb_errors(model, input_model, train_target[b:b + batch_size],
                                                       train_labels[b:b + batch_size], batch_size,
                                                        kernel1, kernel2, kernel3, kconv3)

            loss_per_batches[0, batch] = comparison_error_batch
        loss_per_epochs[e, 0] = torch.mean(loss_per_batches)
    return loss_per_epochs






