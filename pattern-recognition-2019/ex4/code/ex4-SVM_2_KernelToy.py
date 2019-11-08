import matplotlib
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio

from svm import SVM
from plotting_helper import plot_data, plot_kernel_separator

matplotlib.use('TkAgg')

dataPath = '../data/'


def svmKernelToyExample() -> None:
    '''
     - Load non-linear separable toy data_set
     - Train a kernel SVM
     - Print training and test error
     - Plot data and separator
    '''
    data = scio.loadmat(dataPath + 'flower.mat')
    # Only take a subset of the training data
    train = np.append(data['train'][:, :100], data['train'][:, 2100:2200], axis=1)
    d, n = np.shape(train)
    train_x = train[:d - 1, :]
    train_label = np.reshape(train[d - 1, :], (1, n)).astype(float)
    train_label[train_label == 0.0] = -1.0

    d, n = np.shape(data['test'])
    test_x = data['test'][:d - 1, :]
    test_label = np.reshape(data['test'][d - 1, :], (1, n)).astype(float)
    test_label[test_label == 0.0] = -1.0

    plot_data(plt, train_x, train_label, [['red', '+'], ['blue', '_']], 'Training')
    plot_data(plt, test_x, test_label, [['yellow', '+'], ['green', '_']], 'Test')
    plt.show()

    # TODO: Train svm
    svm = ???

    print("Training error")
    # TODO: Compute training error of SVM
    print("Test error")
    # TODO: Compute test error of SVM

    print("Visualizing data")
    # TODO: Visualize data and separation boundary - hint: you can use the given "plot_kernel_separator" and the "plot_data" functions


if __name__ == "__main__":
    print("Python version in use: ", sys.version)
    print("\nSVM exercise - Non-linear Toy Example")
    print("##########-##########-##########")
    svmKernelToyExample()
    print("##########-##########-##########")
