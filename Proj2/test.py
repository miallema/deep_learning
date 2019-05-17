from helpers import generate_disc_set, train_eval
from plotting import plot_learning, plot_points, plot_time, plot_errors
from module import Sequential, Linear, Dropout, ReLU, Tanh
from losses import LossMSE
from optimizers import SGD
from torch import nn, optim, empty


#Model Parameters
rounds = 10
n_epochs = 100
batch_size = 100
num_samples = 1000
learning_rate = 0.001
dropout_p = 0.2

# DIY with dropout and mini batch SGD
errors_diy_dropout = []
times_diy_dropout = []
prediction_diy_dropout = []
losses_diy_dropout = empty(rounds, n_epochs).zero_()

# DIY without dropout and mini batch SGD
errors_diy = []
times_diy = []
prediction_diy = []
losses_diy = empty(rounds, n_epochs).zero_()

# Pytorch with dropout and mini batch SGD
errors_torch = []
times_torch = []
prediction_torch = []
losses_torch = empty(rounds, n_epochs).zero_()

# Pytorch with dropout and Adam
errors_torch_adam = []
times_torch_adam = []
prediction_torch_adam = []
losses_torch_adam = empty(rounds, n_epochs).zero_()

# Accumulation of all 10*1000 generated test samples
x_all = []
for i in range(rounds):

    print('Round:' + str(i+1) + '/' + str(rounds))
    x_train, y_train = generate_disc_set(num_samples)
    x_test, y_test = generate_disc_set(num_samples)
    x_all += list(x_test)

    # ---------------------------------------------------------------------------------------------------------------- #
    # DIY dropout mini batch SGD

    model_diy_dropout = Sequential(Linear(2, 25),
                                ReLU(),
                                Dropout(dropout_p),
                                Linear(25, 25),
                                ReLU(),
                                Dropout(dropout_p),
                                Linear(25, 25),
                                ReLU(),
                                Dropout(dropout_p),
                                Linear(25, 2),
                                Tanh())

    criterion_diy_dropout = LossMSE()
    optimizer_diy_dropout = SGD(model_diy_dropout, learning_rate)

    loss, error, time_taken, prediction = train_eval(n_epochs, model_diy_dropout, criterion_diy_dropout,
                                                     optimizer_diy_dropout, False, x_train, y_train, x_test,
                                                     y_test, batch_size)
    losses_diy_dropout[i] = loss
    times_diy_dropout.append(time_taken)
    errors_diy_dropout.append(error)
    prediction_diy_dropout += list(prediction)

    # ---------------------------------------------------------------------------------------------------------------- #
    # DIY no dropout mini batch SGD

    model_diy = Sequential(Linear(2, 25),
                           ReLU(),
                           Linear(25, 25),
                           ReLU(),
                           Linear(25, 25),
                           ReLU(),
                           Linear(25, 2),
                           Tanh())

    criterion_diy = LossMSE()
    optimizer_diy = SGD(model_diy, learning_rate)

    loss, error, time_taken, prediction = train_eval(n_epochs, model_diy, criterion_diy,
                                                     optimizer_diy, False, x_train, y_train, x_test,
                                                     y_test, batch_size)
    losses_diy[i] = loss
    times_diy.append(time_taken)
    errors_diy.append(error)
    prediction_diy += list(prediction)

    # ---------------------------------------------------------------------------------------------------------------- #
    # Pytorch dropout mini batch SGD
    model_torch = nn.Sequential(nn.Linear(2, 25),
                                nn.ReLU(),
                                nn.Dropout(dropout_p),
                                nn.Linear(25, 25),
                                nn.ReLU(),
                                nn.Dropout(dropout_p),
                                nn.Linear(25, 25),
                                nn.ReLU(),
                                nn.Dropout(dropout_p),
                                nn.Linear(25, 2),
                                nn.Tanh())

    criterion_torch = nn.MSELoss()
    optimizer_torch = optim.SGD(model_torch.parameters(), lr=learning_rate)

    loss, error, time_taken, prediction = train_eval(n_epochs, model_torch, criterion_torch,
                                                     optimizer_torch, True, x_train, y_train, x_test,
                                                     y_test, batch_size)
    losses_torch[i] = loss
    times_torch.append(time_taken)
    errors_torch.append(error)
    prediction_torch += list(prediction)

    # ---------------------------------------------------------------------------------------------------------------- #
    # Pytorch Adam
    model_torch_adam = nn.Sequential(nn.Linear(2, 25),
                                nn.ReLU(),
                                nn.Dropout(dropout_p),
                                nn.Linear(25, 25),
                                nn.ReLU(),
                                nn.Dropout(dropout_p),
                                nn.Linear(25, 25),
                                nn.ReLU(),
                                nn.Dropout(dropout_p),
                                nn.Linear(25, 2),
                                nn.Tanh())

    criterion_torch_adam = nn.MSELoss()
    optimizer_torch_adam = optim.Adam(model_torch_adam.parameters())

    loss, error, time_taken, prediction = train_eval(n_epochs, model_torch_adam, criterion_torch_adam,
                                                     optimizer_torch_adam, True, x_train, y_train, x_test,
                                                     y_test, batch_size)
    losses_torch_adam[i] = loss
    times_torch_adam.append(time_taken)
    errors_torch_adam.append(error)
    prediction_torch_adam += list(prediction)

    # ---------------------------------------------------------------------------------------------------------------- #

#Plots
losses = [losses_diy_dropout, losses_diy, losses_torch, losses_torch_adam]
plot_learning(losses, '../figures/losses_diy.pdf')

plot_points(x_all, prediction_diy_dropout, '../figures/circle_prediction_diy_dropout.pdf')

times = [times_diy, times_diy_dropout, times_torch, times_torch_adam]
plot_time(times, '../figures/boxplot_times.pdf')

errors = [errors_diy, errors_diy_dropout, errors_torch, errors_torch_adam]
plot_errors(errors, '../figures/boxplot_errors.pdf')
