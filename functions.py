import torch
from torch import nn
import dlc_practical_prologue as prologue

#global variables
mini_batch_size = 100
nb_epoch = 25


def error_calculator(target_classes_vect, winner_output, num_samples):
    '''error_calculator takes as input:
    - target_classes_vect, a tensor of the real numbers (100x10 length tensor of zeros and a single one at the position of the number in the second dimension for each batch)
    - winner_output, a tensor of the predicted numbers (100x1 length tensor for each batch)
    - num_samples, the number of samples
    calculates and returns the error on the numbers during prediction.'''
    
    #take the position in the vector that represents the real number of the image
    winner_target=torch.argmax(target_classes_vect, dim=1)
    #Substraction of the predicted number from the real number for all 100 samples
    error_num = winner_output - winner_target
    #Find the number of correct answers where there is no difference between the numbers
    correct = error_num[error_num == 0]
    num_correct = correct.size()
    #find the number of errors
    nb_errors = num_samples[0] - num_correct[0]
    return nb_errors


def compute_nb_errors(model, input_model, target_classes, target_labels, mini_batch_size,
                      kernel1, kernel2, kernel3, kconv3):
    '''compute_nb_errors takes as input: 
    - model, the network that is evaluated
    - input_model, the images given as input to the network
    - target_labels, the labels of images (indicating the comparison of the two images) 
    - target_classes classes (indicating the number represented by each image)
    - mini_batch_size, an integer defining the number of samples being, in each subset of samples, given to train the network
    - the sizes of the kernels, kernel1/2/3
    - and the number of filters of the third convolution
    computes and returns losses on numbers (with auxilliary) and comparison.'''
    
    #initialization of the counters of errors
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

    #Calculation of proportions of errors
    errors_comparison = errors_comparison * 100 / 1000
    errors_num0 = errors_num0 * 100 / 1000
    errors_num1 = errors_num1 * 100 / 1000

    return errors_num0, errors_num1, errors_comparison




def train_model(model, auxiliary, train_input, train_classes, train_labels, nb_epochs, eta, \
                kernel1, kernel2, kernel3, kconv3):
    '''train_model takes as input :
    -model, the network to be trained
    -auxiliary, boolean variable indicating if we consider an auxilliary loss or not
    -train_input, tensors of 1000 pairs of images each
    -train_classes,  a 1000x2 tensor indicating the number represented by each image
    -train_labels, a 1000x1 tensor indicating the number represented by each image
    -nb_epochs, the number of time that the trainset goes through the network
    -eta, the size of the learning step
    trains the model and returns the losses per number of epoch.'''
    
    #define criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=eta)
    batch_size = 100

    #initialize shells 
    loss_per_epochs = torch.zeros([nb_epochs, 1])
    
    for e in range(nb_epochs):
        #initialize for every epoch a shell
        loss_per_batches= torch.zeros([1, len(range(0, train_input.size(0), batch_size))+1])
        batch = 0

        for b in range(0, train_input.size(0), batch_size):

            #take a subset of the train to be passed to the network
            input_model = train_input[b: b + batch_size]

            #Set the model in train mode to do a dropout and batchnormalization
            model.train()
            output_model = model(input_model, kernel1, kernel2, kernel3, kconv3)

            #calculate the cross entropy loss
            loss_comparison = criterion(output_model[:, 0:2], train_labels[b:b + batch_size])
            loss_number1 = criterion(output_model[:, 2:12], train_classes[b:b + batch_size][:, 0])
            loss_number2 = criterion(output_model[:, 12:22], train_classes[b:b + batch_size][:, 1])

            #Ponder the auxilliary loss
            if auxiliary == True:
                loss = 1 / 2 * loss_comparison + 1 / 4 * loss_number1 + 1 / 4 * loss_number2
            else:
                loss = loss_comparison

    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch += 1

            comparison_error_batch, _, _ = compute_nb_errors(model, input_model, train_classes[b:b + batch_size],
                                                       train_labels[b:b + batch_size], batch_size,
                                                        kernel1, kernel2, kernel3, kconv3)

            loss_per_batches[0, batch] = comparison_error_batch
        loss_per_epochs[e, 0] = torch.mean(loss_per_batches)
    return loss_per_epochs






