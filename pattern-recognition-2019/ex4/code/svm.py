import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm
import cvxopt as cvx


class SVM(object):
    '''
    SVM class
    '''

    def __init__(self, C=None):
        self.C = C
        self.__TOL = 1e-5

    def __linearKernel__(self, x1: np.ndarray, x2: np.ndarray, _) -> float:
        # TODO: Implement linear kernel function
        # @x1 and @x2 are vectors
        return np.dot(x1.T, x2)

    def __polynomialKernel__(self, x1: np.ndarray, x2: np.ndarray, p: int) -> float:
        # TODO: Implement polynomial kernel function
        # @x1 and @x2 are vectors
        return (np.dot(x1.T, x2) + 1) ** p

    def __gaussianKernel__(self, x1: np.ndarray, x2: np.ndarray, sigma: float) -> float:
        # TODO: Implement gaussian kernel function
        # @x1 and @x2 are vectors
        return np.exp(- (norm(x1 - x2) ** 2) / (2 * (sigma ** 2)))

    def __computeKernelMatrix__(self, x: np.ndarray, kernelFunction, pars) -> np.ndarray:
        # TODO: Implement function to compute the kernel matrix
        # @x is the data matrix
        # @kernelFunction - pass a kernel function (gauss, poly, linear) to this input
        # @pars - pass the possible kernel function parameter to this input
        n, m = x.shape
        K = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                K[i, j] = kernelFunction(x[:, i], x[:, j], pars)
        return K

    def train(self, x: np.ndarray, y: np.ndarray, kernel=None, kernelpar=2) -> None:

        # TODO: Implement the remainder of the svm training function
        self.kernelpar = kernelpar

        NUM = x.shape[1]

        # we'll solve the dual
        # obtain the kernel
        if kernel == 'linear':
            # TODO: Compute the kernel matrix for the non-linear SVM with a linear kernel
            print('Fitting SVM with linear kernel')
            K = self.__computeKernelMatrix__(x, self.__linearKernel__, None)
            self.kernel = self.__linearKernel__
        elif kernel == 'poly':
            # TODO: Compute the kernel matrix for the non-linear SVM with a polynomial kernel
            print('Fitting SVM with Polynomial kernel, order: {}'.format(kernelpar))
            K = self.__computeKernelMatrix__(x, self.__polynomialKernel__, kernelpar)
        elif kernel == 'rbf':
            # TODO: Compute the kernel matrix for the non-linear SVM with an RBF kernel
            print('Fitting SVM with RBF kernel, sigma: {}'.format(kernelpar))
            K = self.__computeKernelMatrix__(x, self.__gaussianKernel__, kernelpar)
        else:
            print('Fitting linear SVM')
            # TODO: Compute the kernel matrix for the linear SVM
            K = self.__computeKernelMatrix__(x, self.__linearKernel__, None)

        if self.C is None:
            G = cvx.matrix(-np.eye(NUM))
            h = cvx.matrix(np.zeros(NUM))
        else:
            print("Using Slack variables")
            identity_matrix = np.eye(NUM)
            G = cvx.matrix(np.vstack((-identity_matrix, identity_matrix)))
            h = cvx.matrix(np.hstack((np.zeros(NUM), np.ones(NUM) * self.C)))
        self.k = K
        P = cvx.matrix(np.outer(y, y) * K)
        q = cvx.matrix(-np.ones((NUM, 1)))
        b = cvx.matrix(0.0)
        A = cvx.matrix(y)
        solution = cvx.solvers.qp(P, q, G, h, A, b)
        lambdas = np.ravel(solution['x'])

        # TODO: Compute below values according to the lecture slides
        self.lambdas = lambdas[lambdas > self.__TOL]  # Only save > 0
        indexes_sv = np.ravel(np.argwhere(lambdas > self.__TOL))
        self.sv = x[:, indexes_sv]  # List of support vectors
        self.sv_labels = y[0, indexes_sv]  # List of labels for the support vectors (-1 or 1 for each support vector)
        self.w = 0
        if kernel is None:
            for i in range(self.sv.shape[1]):
                self.w += self.lambdas[i] * self.sv_labels[i] * self.sv[:, i]  # SVM weights used in the linear SVM
            # Use the mean of all support vectors for stability when computing the bias (w_0)
        else:
            for i in range(self.sv.shape[1]):
                self.w += self.lambdas[i] * self.sv_labels[i] * self.k[:, i]
            # Use the mean of all support vectors for stability when computing the bias (w_0).
            # In the kernel case, remember to compute the inner product with the chosen kernel function.
        mean_sv = 0
        for i in range(self.lambdas.shape[0]):
            mean_sv += self.sv[:, i]
        mean_sv = np.array([mean_sv / self.lambdas.shape[0]]).T
        m = self.lambdas * self.sv_labels
        self.bias = self.sv_labels[0] - np.dot(np.array(self.w).T, mean_sv)  # Bias

        # TODO: Implement the KKT check
        self.__check__()

    def __check__(self) -> None:
        # Checking implementation according to KKT2 (Linear_classifiers slide 46)
        kkt2_check = np.dot(self.lambdas, self.sv_labels)
        assert kkt2_check < self.__TOL, 'SVM check failed - KKT2 condition not satisfied'

    def classifyLinear(self, x: np.ndarray) -> np.ndarray:
        '''
        Classify data given the trained linear SVM - access the SVM parameters through self.
        :param x: Data to be classified
        :return: List of classification values (-1.0 or 1.0)
        '''
        # TODO: Implement
        classifications = np.zeros(x.shape[1])
        for i in range(x.shape[1]):
            g = np.dot(self.w.T, x[:, i]) + self.bias
            classifications[i] = np.sign(g)
        return classifications

    def printLinearClassificationError(self, x: np.ndarray, y: np.ndarray) -> None:
        '''
        Calls classifyLinear and computes the total classification error given the ground truth labels
        :param x: Data to be classified
        :param y: Ground truth labels
        '''
        # TODO: Implement
        labels = self.classifyLinear(x)
        result = np.count_nonzero(labels - y)
        print("Total error: {:.2f}%".format(result))

    def classifyKernel(self, x: np.ndarray) -> np.ndarray:
        '''
        Classify data given the trained kernel SVM - use self.kernel and self.kernelpar to access the kernel function and parameter
        :param x: Data to be classified
        :return: List of classification values (-1.0 or 1.0)
        '''
        g = np.dot(np.dot(self.lambdas, self.sv_labels), self.k) + self.bias
        classifications = []
        for i in range(g.shape[1]):
            classifications.append(np.sign(g[0, i]))
        return classifications

    def printKernelClassificationError(self, x: np.ndarray, y: np.ndarray) -> None:
        '''
        Calls classifyKernel and computes the total classification error given the ground truth labels
        :param x: Data to be classified
        :param y: Ground truth labels
        '''
        # TODO: Implement
        labels = self.classifyKernel(x)
        result = len(np.count_nonzero(labels - y))
        print("Total error: {:.2f}%".format(result))
