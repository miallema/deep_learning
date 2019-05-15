import torch
import model_no_sharedweights as mnsw
import model_sharedweights as msw
import functions as fc
from time import time

def hyperparam_tuning(model_ID, auxiliary, train_input, train_classes, train_labels, test_input, test_classes, test_labels):
    '''The function hyperparam_tuning takes as input : 
    - a string (model_ID) indicating if the model shared weights or not
    - a boolean variable, auxiliary, indicating if we consider an auxilliary loss or not
    - a train and test tensors (1000 pairs of images each) as input
    - their labels (indicating the comparison of the two images, a 1000x1 tensor) 
    - and classes (indicating the number represented by each image, a 1000x2 tensor)
    The function perfoms an hyperparameter tuning for the type of model considered and returns a tensor referencing all errors for each combination of hyperparameters defined in the function.'''

    #hyperparameters to be tuned
    #Proportions of dropout
    probas = [0, 0.25, 0.5]
    #numbers of filters for the convolutions
    kconvs1 = [10, 30, 60]
    kconvs2 = [10, 30, 60]
    kconvs3 = [10, 30, 60]
    #size of the squared kernel 
    kernels1 = [3, 5]
    kernels2 = [2, 3]
    kernels3 = [2, 3]
    #Size of learning step
    etas = [0.100, 0.010, 0.001]
    

    #number of combinations of hyperparameters explored
    nb_combos = len(probas) * len(kconvs1) * len(kconvs2) * len(kconvs1) * len(kernels1) * len(kernels1) * len(
        kernels1) * len(etas)
    print("Number of model combinations explored during hyperparameter optimization: %d" %(nb_combos))
    
    #9 Hyperparameters and 6 Accuracies for each combo
    hyperparam_tensor = torch.zeros([nb_combos, 9 + 6], dtype=torch.float32)
    nb_model = 0

    #Tuning
    for kconv1 in kconvs1:
        for kconv2 in kconvs2:
            for kconv3 in kconvs3:
                for kernel1 in kernels1:
                    for kernel2 in kernels2:
                        for kernel3 in kernels3:
                            for proba in probas:
                                for eta in etas:
                                    
                                    #initialization of the Networks
                                    if model_ID == 'WeightSharing':
                                        model = msw.Net_SharedWeights(proba, kconv1, kconv2, kconv3, kernel1,
                                                             kernel2, kernel3)
                                    else:
                                        model = mnsw.Net_NoSharedWeights(proba, kconv1, kconv2, kconv3,
                                                                         kernel1, kernel2, kernel3)

                                    #training of the models
                                    fc.train_model(model, auxiliary, train_input, train_classes, train_labels, fc.nb_epoch, \
                                                eta, kernel1, kernel2, kernel3, kconv3)
                                    
                                    # Calculate Train and Test error percentages
                                    #Train
                                    errors_num0_train, errors_num1_train, errors_comparison_train = \
                                        fc.compute_nb_errors(model, train_input, train_classes, \
                                                          train_labels, fc.mini_batch_size, \
                                                          kernel1, kernel2, kernel3, kconv3)
                                    #Test
                                    errors_num0_test, errors_num1_test, errors_comparison_test = \
                                        fc.compute_nb_errors(model, test_input, test_classes, \
                                                          test_labels, fc.mini_batch_size, \
                                                          kernel1, kernel2, kernel3, kconv3)

                                    #Save errors for the actual combo of hyperparameters
                                    hyperparam_tensor[nb_model, :] = torch.Tensor([kconv1, kconv2, kconv3, \
                                                                                   kernel1, kernel2, kernel3, \
                                                                                   proba, eta, fc.nb_epoch, \
                                                                                   errors_comparison_test,
                                                                                   errors_num0_test, errors_num1_test, \
                                                                                   errors_comparison_train,
                                                                                   errors_num0_train,
                                                                                   errors_num1_train])
                                    nb_model += 1
    print('Done')
    return hyperparam_tensor



def model_performance_eval(model_ID, train_input, train_classes, train_labels, \
                           test_input, test_classes, test_labels,\
                           auxiliary, final_proba, final_kconv1, final_kconv2, final_kconv3, \
                           final_kernel1, final_kernel2, final_kernel3, final_eta):
    '''The function model_performance_eval takes as input:
    - a string (model_ID) indicating if the model shared weights or not
    - a boolean variable, auxiliary, indicating if we consider an auxilliary loss or not
    - a train and test tensors (1000 pairs of images each) as input
    - their labels (indicating the comparison of the two images, a 1000x1 tensor) 
    - their classes (indicating the number of each image, a 1000x2) 
    - and all tuned hyperparameters needed for the model 
    The function performs 10 rounds in order to evaluate the model and returns: 
    - a tensor with train and test errors for the approximation of the two numbers and their comparison 
    - a tensor of the losses per epochs'''
    
    #empty shells for errors: 6 accuracies to save for 10 rounds
    errors_tensor = torch.zeros([10,6], dtype=torch.float32)
    rounds = range(10)

    loss_per_round = torch.zeros([len(rounds), fc.nb_epoch])
    for r in rounds:
        #shuffle the data
        size_train=train_input.size()
        ind_shuffled=torch.randperm(size_train[0])
        train_input_shuffled=train_input[ind_shuffled]
        train_classes_shuffled=train_classes[ind_shuffled]
        train_labels_shuffled=train_labels[ind_shuffled]

        #Create the Nets with the tuned hyperparameters
        if model_ID == 'WeightSharing':
            final_model = msw.Net_SharedWeights(final_proba, final_kconv1, final_kconv2, final_kconv3, \
                                                   final_kernel1, final_kernel2, final_kernel3)
        else:
            final_model = mnsw.Net_NoSharedWeights(final_proba, final_kconv1, final_kconv2, final_kconv3, \
                                                   final_kernel1, final_kernel2, final_kernel3)

        start = time()
        loss_per_epochs = fc.train_model(final_model, auxiliary, train_input_shuffled, train_classes_shuffled, train_labels_shuffled, fc.nb_epoch, \
                                                        final_eta, final_kernel1, final_kernel2, final_kernel3, \
                                                        final_kconv3)
        end = time ()
        
        num_parameters = sum(p.numel() for p in final_model.parameters())
       
        #Calculate Train and Test error percentages
        #Train
        errors_num0_train, errors_num1_train, errors_comparison_train = \
                fc.compute_nb_errors(final_model, train_input, train_classes, \
                                  train_labels, fc.mini_batch_size, \
                                  final_kernel1, final_kernel2, final_kernel3, final_kconv3)

        #Test
        errors_num0_test, errors_num1_test, errors_comparison_test = \
                fc.compute_nb_errors(final_model, test_input, test_classes, \
                                  test_labels, fc.mini_batch_size, \
                                  final_kernel1, final_kernel2, final_kernel3, final_kconv3)

        #Stock errors
        errors_tensor[r, :] = torch.Tensor([errors_comparison_test,errors_num0_test,errors_num1_test,\
                                        errors_comparison_train, errors_num0_train, errors_num1_train])
        loss_per_round[r, :] = loss_per_epochs.squeeze()

    print('time needed to train the current model : ', end-start, ' s ')
    print('number of parameters of the model : ', num_parameters)
           
    return errors_tensor, loss_per_round