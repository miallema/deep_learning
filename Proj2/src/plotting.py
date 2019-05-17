from torch import mean, std
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


#Plot the loss over 10 rounds keeping track of the mean and std. 
def plot_learning(losses, fname):
    labels_mean = ['Mean diy dropout',
                   'Mean diy no dropout',
                   'Mean torch dropout SGD',
                   'Mean torch dropout adam']
    labels_std = ['Std diy dropout',
                  'Std diy no dropout',
                  'Std torch dropout SGD',
                  'Std torch dropout adam']
    colors = ['green', 'blue', 'yellow', 'red']
    r = np.shape(losses[0])[1]
    plt.figure(figsize=[12, 7])
    for i in range(len(losses)):
        m = mean(losses[i], dim=0).numpy()
        s = std(losses[i], dim=0).numpy()
        plt.plot(range(r), m, label=labels_mean[i], color=colors[i])
        plt.fill_between(range(r), [x if x > 0 else 0 for x in (m - s)], m + s,
                         color=colors[i], alpha=0.2, label=labels_std[i])
    plt.legend(fontsize=16)
    plt.ylabel('MSE Loss', fontsize=20)
    plt.xlabel('Epoch', fontsize=20)
    plt.savefig(fname)


#Plot the predictions in the 2-d plane with colorcode according to label.
def plot_points(x_test, prediction, fname):
    x_test = zip(*x_test)
    x_test = [list(x) for x in x_test]
    x = [x.item() for x in x_test[0]]
    y = [y.item() for y in x_test[1]]
    plt.figure(figsize=(10, 10))
    plt.scatter(x, y, c=prediction, cmap=matplotlib.colors.ListedColormap(['blue', 'red']))
    plt.savefig(fname)


#Boxplots of time
def plot_time(times, fname):
    plt.figure(figsize=(12, 7))
    plt.boxplot(times)
    x = np.array([1, 2, 3, 4])
    my_xticks = ['diy SGD', 'diy dropout SGD', 'torch dropout SGD', 'torch dropout Adam']
    plt.xticks(x, my_xticks, fontsize=16)
    plt.ylabel('Time[s]', fontsize=20)
    plt.savefig(fname)


#Boxplots of errors
def plot_errors(errors, fname):
    plt.figure(figsize=(12, 7))
    plt.boxplot(errors)
    x = np.array([1, 2, 3, 4])
    my_xticks = ['diy SGD', 'diy dropout SGD', 'torch dropout SGD', 'torch dropout Adam']
    plt.xticks(x, my_xticks, fontsize=16)
    plt.ylabel('Test errors[%]', fontsize=20)
    plt.savefig(fname)



