import torch

#for plotting only:
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def plot_results(error_tensor_aux, error_tensor, model_ID):
    # convert to numpy matrices for an easier handling
    error_matrix_aux = error_tensor_aux.numpy()
    error_matrix = error_tensor.numpy()

    # extract the desired information
    errors_test = np.concatenate((error_matrix_aux[:, 0:3], error_matrix[:, 0:3]), axis=1)
    errors_test_train = np.stack((error_matrix_aux[:, 0], error_matrix_aux[:, 3], \
                                  error_matrix[:, 0], error_matrix[:, 3]), axis=1)

    print('Without Auxiliary: Mean Error Comparison for Test set is {} %, and for Train set is {} %'
          .format(np.mean(error_matrix[:, 0]), np.mean(error_matrix[:, 3])))

    print('With Auxiliary: Mean Error Comparison for Test set is {} %, and for Train set is {} %'
          .format(np.mean(error_matrix_aux[:, 0]), np.mean(error_matrix_aux[:, 3])))
    print('With Auxiliary: Mean Error Number 1 for Test set is {} %, and for Number2 Test set is {} %'
          .format(np.mean(error_matrix_aux[:, 1]), np.mean(error_matrix_aux[:, 2])))

    # formatting of the texts on the images
    matplotlib.rc('xtick', labelsize=16)
    matplotlib.rc('ytick', labelsize=16)

    # plot the test error on the comparison (with and without auxiliary losses)
    matplotlib.rc('xtick', labelsize=16)
    matplotlib.rc('ytick', labelsize=16)


    plt.figure(figsize=(15, 7))
    plt.boxplot(errors_test[:, 0:4])
    x = np.array([1, 2, 3, 4])
    my_xticks = ['Comparison [Aux]', 'Number1 [Aux]', 'Number2[Aux]', 'Comparison']
    plt.xticks(x, my_xticks)
    plt.ylabel('Percentage of Test Error (%)', fontsize=16)
    plt.title('%s Comparison Test Error with and Without Auxiliary Losses (on Number1 and Number2)' %(model_ID),
              fontsize='16')
    plt.savefig('figures/%s Comparison_test_error.png' %(model_ID))

    # plot the comparison error for test and train (with and without auxiliary losses)
    plt.figure(figsize=(15, 7))
    plt.boxplot(errors_test_train)
    x = np.array([1, 2, 3, 4])
    my_xticks = ['Test [Aux]', 'Train [Aux]', 'Test', 'Train']
    plt.xticks(x, my_xticks)
    plt.ylabel('Percentage of Error (%)', fontsize=16)
    plt.title('%s Train and Test Comparison Errors With and Without Auxiliary Losses (on Number1 and Number2)'%(model_ID),
              fontsize='16')
    plt.savefig('figures/%s train_test.png' %(model_ID))

def plot_learning(loss_per_round_aux, loss_per_round, model_ID):

    losses_aux = torch.mean(loss_per_round_aux, dim=0)
    losses = torch.mean(loss_per_round, dim=0)

    losses_aux_std = torch.std(loss_per_round_aux, dim=0)
    losses_std = torch.std(loss_per_round, dim=0)

    mean_aux = losses_aux.detach().numpy()
    mean = losses.detach().numpy()
    std_aux = losses_aux_std.detach().numpy()
    std = losses_std.detach().numpy()

    # formatting of the texts on the images
    matplotlib.rc('xtick', labelsize=16)
    matplotlib.rc('ytick', labelsize=16)

    plt.figure(figsize=(15, 7))
    plt.plot(range(1, 26), mean_aux, 'b-', label='WITH Auxiliary Losses')
    plt.plot(range(1, 26), mean_aux, 'ro')
    plt.fill_between(range(1, 26), mean_aux - std_aux, mean_aux + std_aux,
                     color='blue', alpha=0.2, label='Standard deviation')
    plt.plot(range(1, 26), mean,'ro')
    plt.plot(range(1, 26), mean, 'g-', label='WITHOUT Auxiliary Losses')
    plt.fill_between(range(1, 26), mean - std, mean + std,
                     color='green', alpha=0.2, label='Standard deviation')
    plt.legend()
    plt.ylabel('Comparison Error (%)', fontsize=16)
    plt.xlabel('Number of Epochs')
    plt.xlim(left=1, right=25)
    plt.title('%s Learning Rates per Epochs during Training (Averaged over Batches and 10+ Rounds)' %(model_ID),
              fontsize='16')
    plt.savefig('figures/%s Learning Rates' %(model_ID))