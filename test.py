import torch
import dlc_practical_prologue as prologue
import processing as process
import plotting as pl

if __name__ == '__main__' :

    mini_batch_size = 100
    nb_epoch = 25

    ## Create samples
    nb = 1000
    train_input, train_labels, train_classes, test_input, test_labels, test_classes = prologue.generate_pair_sets(nb)
    _, _, _, validation_input, validation_labels, validation_classes = prologue.generate_pair_sets(nb)

    model_IDs = ['WeightSharing', 'No WeightSharing']

    for model_ID in model_IDs:
        print(model_ID)

        ## Hyperparameter tuning
        # with auxiliary
        auxiliary = True
        hyperparam_tensor_aux = process.hyperparam_tuning(model_ID, auxiliary, train_input, train_classes, train_labels,\
                                                    validation_input, validation_classes, validation_labels)
        # without auxiliary
        auxiliary = False
        hyperparam_tensor = process.hyperparam_tuning(model_ID, auxiliary, train_input, train_classes, train_labels,\
                                                validation_input, validation_classes, validation_labels)

        ## Get best parameters for our model
        #with auxiliary
        value_aux, index_aux = torch.min(hyperparam_tensor_aux[:, 9], 0)
        print('Parameters with auxiliary loss:')
        print(hyperparam_tensor_aux[index_aux, :])
        #without
        value, index = torch.min(hyperparam_tensor[:, 9], 0)
        print('Parameters without auxiliary loss:')
        print(hyperparam_tensor[index, :])

        ## 10 rounds to evaluate our model

        #with auxiliary
        auxiliary = True
        final_kconv1_aux = int(hyperparam_tensor_aux[index_aux, 0].item())
        final_kconv2_aux = int(hyperparam_tensor_aux[index_aux, 1].item())
        final_kconv3_aux = int(hyperparam_tensor_aux[index_aux, 2].item())
        final_kernel1_aux = int(hyperparam_tensor_aux[index_aux, 3].item())
        final_kernel2_aux = int(hyperparam_tensor_aux[index_aux, 4].item())
        final_kernel3_aux = int(hyperparam_tensor_aux[index_aux, 5].item())
        final_proba_aux = hyperparam_tensor_aux[index_aux, 6].item()
        final_eta_aux = hyperparam_tensor_aux[index_aux, 7].item()

        # Evaluation of the final model
        error_tensor_aux, loss_per_round_aux = process.model_performance_eval(model_ID, train_input, train_classes, train_labels, \
                                                    test_input, test_classes, test_labels, auxiliary,\
                                                    final_proba_aux, final_kconv1_aux, final_kconv2_aux,\
                                                    final_kconv3_aux, final_kernel1_aux, final_kernel2_aux, \
                                                    final_kernel3_aux, final_eta_aux)

        #without auxiliary
        auxiliary = False
        final_kconv1 = int(hyperparam_tensor[index, 0].item())
        final_kconv2 = int(hyperparam_tensor[index, 1].item())
        final_kconv3 = int(hyperparam_tensor[index, 2].item())
        final_kernel1 = int(hyperparam_tensor[index, 3].item())
        final_kernel2 = int(hyperparam_tensor[index, 4].item())
        final_kernel3 = int(hyperparam_tensor[index, 5].item())
        final_proba = hyperparam_tensor[index, 6].item()
        final_eta = hyperparam_tensor[index, 7].item()

        # Evaluation of the final model
        error_tensor, loss_per_round = process.model_performance_eval(model_ID, train_input, train_classes, train_labels, \
                                                test_input, test_classes, test_labels, auxiliary, \
                                                final_proba, final_kconv1, final_kconv2, final_kconv3, \
                                                final_kernel1, final_kernel2, final_kernel3, final_eta)


        #plot results
        pl.plot_results(error_tensor_aux, error_tensor, model_ID)
        pl.plot_learning(loss_per_round_aux, loss_per_round, model_ID)
