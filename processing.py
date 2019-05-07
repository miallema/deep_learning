import torch
import model_no_sharedweights as mnsw
import model_sharedweights as msw
import functions as fc

def hyperparam_tuning(model_ID, auxiliary, train_input, train_classes, train_labels, test_input, test_classes, test_labels):
    random_init = False
    probas = [0, 0.25, 0.5]
    kconvs1 = [10, 30, 60]
    kconvs2 = [10, 30, 60]
    kconvs3 = [10, 30, 60]
    kernels1 = [3, 5]
    kernels2 = [2, 3]
    kernels3 = [2, 3]
    etas = [0.100, 0.010, 0.001]

    #miniTEST
    probas = [0]
    kconvs1 = [10]
    kconvs2 = [10]
    kconvs3 = [10]
    kernels1 = [3]
    kernels2 = [2]
    kernels3 = [2]
    etas = [0.100]

    nb_combos = len(probas) * len(kconvs1) * len(kconvs2) * len(kconvs1) * len(kernels1) * len(kernels1) * len(
        kernels1) * len(etas)
    print("Number of model combinations explored during hyperparameter optimization: %d" %(nb_combos))
    # 9 Hyperparameters and 6 Accuracies for each combo
    hyperparam_tensor = torch.zeros([nb_combos, 9 + 6], dtype=torch.float32)
    nb_model = 0

    for kconv1 in kconvs1:

        for kconv2 in kconvs2:

            for kconv3 in kconvs3:

                for kernel1 in kernels1:

                    for kernel2 in kernels2:

                        for kernel3 in kernels3:

                            for proba in probas:

                                for eta in etas:

                                    if model_ID == 'WeightSharing':
                                        model = msw.Net_SharedWeights(random_init, proba, kconv1, kconv2, kconv3, kernel1,
                                                             kernel2, kernel3)
                                    else:
                                        model = mnsw.Net_NoSharedWeights(random_init, proba, kconv1, kconv2, kconv3,
                                                                         kernel1, kernel2, kernel3)



                                    fc.train_model(model, auxiliary, train_input, train_classes, train_labels, fc.nb_epoch, \
                                                eta, kernel1, kernel2, kernel3, kconv3)

                                    # Calculate Train and Test error percentages
                                    errors_num0_train, errors_num1_train, errors_comparison_train = \
                                        fc.compute_nb_errors(model, train_input, train_classes, \
                                                          train_labels, fc.mini_batch_size, \
                                                          kernel1, kernel2, kernel3, kconv3)

                                    errors_num0_test, errors_num1_test, errors_comparison_test = \
                                        fc.compute_nb_errors(model, test_input, test_classes, \
                                                          test_labels, fc.mini_batch_size, \
                                                          kernel1, kernel2, kernel3, kconv3)

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
    #initalize weights to random values and initializ biases
    #we tried this random initialization but it yielded no improvement. We leave the possibility to try it out here.
    random_init = False

    #empty shells for errors: 6 accuracies to save for 10 rounds
    errors_tensor = torch.zeros([10,6], dtype=torch.float32)
    rounds = range(10)

    #miniTEST
    rounds =range(2)

    loss_per_round = torch.zeros([len(rounds), fc.nb_epoch])
    for r in rounds:
        #shuffle the data
        size_train=train_input.size()
        ind_shuffled=torch.randperm(size_train[0])
        train_input_shuffled=train_input[ind_shuffled]
        train_classes_shuffled=train_classes[ind_shuffled]
        train_labels_shuffled=train_labels[ind_shuffled]

        if model_ID == 'WeightSharing':
            final_model = msw.Net_SharedWeights(random_init, final_proba, final_kconv1, final_kconv2, final_kconv3, \
                                                   final_kernel1, final_kernel2, final_kernel3)
        else:
            final_model = mnsw.Net_NoSharedWeights(random_init, final_proba, final_kconv1, final_kconv2, final_kconv3, \
                                                   final_kernel1, final_kernel2, final_kernel3)



        loss_per_epochs = fc.train_model(final_model, auxiliary, train_input_shuffled, train_classes_shuffled, train_labels_shuffled, fc.nb_epoch, \
                                                        final_eta, final_kernel1, final_kernel2, final_kernel3, \
                                                        final_kconv3)

        #Calculate Train and Test error percentages
        errors_num0_train, errors_num1_train, errors_comparison_train = \
                fc.compute_nb_errors(final_model, train_input, train_classes, \
                                  train_labels, fc.mini_batch_size, \
                                  final_kernel1, final_kernel2, final_kernel3, final_kconv3)

        errors_num0_test, errors_num1_test, errors_comparison_test = \
                fc.compute_nb_errors(final_model, test_input, test_classes, \
                                  test_labels, fc.mini_batch_size, \
                                  final_kernel1, final_kernel2, final_kernel3, final_kconv3)

        errors_tensor[r, :] = torch.Tensor([errors_comparison_test,errors_num0_test,errors_num1_test,\
                                        errors_comparison_train, errors_num0_train, errors_num1_train])

        loss_per_round[r, :] = loss_per_epochs.squeeze()

    return errors_tensor, loss_per_round