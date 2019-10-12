import sys, os, math
import random
import scipy.io
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from numpy.linalg import inv
from matplotlib.patches import Ellipse
from imageHelper import imageHelper
from myMVND import MVND
from classifyHelper import classify, get_prior

matplotlib.use('TkAgg')

dataPath = '../data/'


def gmm_draw(gmm, data, plotname='') -> None:
    '''
    gmm helper function to visualize cluster assignment of data
    :param gmm:         list of MVND objects
    :param data:        Training inputs, #(dims) x #(samples)
    :param plotname:    Optional figure name
    '''
    plt.figure(plotname)
    K = len(gmm)
    N = data.shape[1]
    dists = np.zeros((K, N))
    for k in range(0, K):
        d = data - (np.kron(np.ones((N, 1)), gmm[k].mean)).T
        dists[k, :] = np.sum(np.multiply(np.matmul(inv(gmm[k].cov), d), d), axis=0)
    comp = np.argmin(dists, axis=0)

    # plot the input data
    ax = plt.gca()
    ax.axis('equal')
    for (k, g) in enumerate(gmm):
        indexes = np.where(comp == k)[0]
        kdata = data[:, indexes]
        g.data = kdata
        ax.scatter(kdata[0, :], kdata[1, :])

        [_, L, V] = scipy.linalg.svd(g.cov, full_matrices=False)
        phi = math.acos(V[0, 0])
        if float(V[1, 0]) < 0.0:
            phi = 2 * math.pi - phi
        phi = 360 - (phi * 180 / math.pi)
        center = np.array(g.mean).reshape(1, -1)

        d1 = 2 * np.sqrt(L[0])
        d2 = 2 * np.sqrt(L[1])
        ax.add_patch(Ellipse(center.T, d1, d2, phi, fc='#CCCCCC', lw=3, alpha=0.5, zorder=1, fill=False))
        plt.plot(center[0, 0], center[0, 1], 'kx')


def gmm_em(data, K: int, iter: int, plot=False) -> list:
    '''
    EM-algorithm for Gaussian Mixture Models
    Usage: gmm = gmm_em(data, K, iter)
    :param data:    Training inputs, #(dims) x #(samples)
    :param K:       Number of GMM components, integer (>=1)
    :param iter:    Number of iterations, integer (>=0)
    :param plot:    Enable/disable debugging plotting
    :return:        List of objects holding the GMM parameters.
                    Use gmm[i].mean, gmm[i].cov, gmm[i].c
    '''
    eps = sys.float_info.epsilon
    [d, N] = data.shape
    gmm = []
    # TODO: EXERCISE 2 - Implement E and M step of GMM algorithm
    # Hint - first randomly assign a cluster to each sample
    mixing_coeff = 1 / K
    # clusters is an array where each entry maps a the cluster number for each data sample
    clusters = np.zeros((N,))
    for sample in range(N):
        clusters[sample] = np.random.random_integers(1, K)
    # create an array with data for each randomly assigned cluster
    for cluster_no in range(1, K + 1):
        assigned_indeces = np.where(clusters == cluster_no)[0]
        cluster_size = len(assigned_indeces)
        data_for_cluster = np.zeros((d, cluster_size))
        for ind in range(cluster_size):
            data_for_cluster[:, ind] = data[:, assigned_indeces[ind]]
        gmm.append(MVND(data_for_cluster, mixing_coeff))
    # Hint - then iteratively update mean, cov and p value of each cluster via EM
    # Hint - use the gmm_draw() function to visualize each step
    for it in range(iter):
        gmm_draw(gmm, data, 'iteration ' + str(it))
        new_clusters = []
        # cluster_responsibilities for calculating c_new
        cluster_responsibilities = []
        # initialize empty data for new clusters
        for component in range(K):
            new_clusters.insert(component, [])
            cluster_responsibilities.insert(component, [])
        for sample in range(N):
            # calculate responsibility
            p = 0
            responsibility = []
            for component in range(K):
                cluster_mvnd = gmm[component]
                p += cluster_mvnd.c * cluster_mvnd.pdf(data[:, sample])
            # find max responsibility
            for component in range(K):
                cluster_mvnd = gmm[component]
                p_component = (cluster_mvnd.c * cluster_mvnd.pdf(data[:, sample])) / p
                responsibility.append(p_component)
            max_cluster = np.argmax(responsibility, 0)
            # assign sample to maximum responsibility
            new_clusters[max_cluster].append(data[:, sample])
            cluster_responsibilities[max_cluster].append(np.amax(responsibility))
            # check for convergence
        # make new gmm
        gmm.clear()
        for component in range(K):
            c_new = np.sum(cluster_responsibilities[component]) / N
            gmm.append(MVND(np.transpose(np.asarray(new_clusters[component])), c_new))
        plt.draw()
        plt.pause(0.001)
    return gmm


def gmmToyExample() -> None:
    '''
    GMM toy example - load toyexample data and visualize cluster assignment of each datapoint
    '''
    gmmdata = scipy.io.loadmat(os.path.join(dataPath, 'gmmdata.mat'))['gmmdata']
    gmm_em(gmmdata, 3, 20, plot=True)


def gmmSkinDetection() -> None:
    '''
    Skin detection - train a GMM for both skin and non-skin.
    Classify the test and training image using the classify helper function.
    Note that the "mask" binary images are used as the ground truth.
    '''
    K = 3
    iter = 10
    sdata = scipy.io.loadmat(os.path.join(dataPath, 'skin.mat'))['sdata']
    ndata = scipy.io.loadmat(os.path.join(dataPath, 'nonskin.mat'))['ndata']
    gmms = gmm_em(sdata, K, iter)
    gmmn = gmm_em(ndata, K, iter)

    print("TRAINING DATA")
    trainingmaskObj = imageHelper()
    trainingmaskObj.loadImageFromFile(os.path.join(dataPath, 'mask.png'))
    trainingimageObj = imageHelper()
    trainingimageObj.loadImageFromFile(os.path.join(dataPath, 'image.png'))
    prior_skin, prior_nonskin = get_prior(trainingmaskObj)
    classify(trainingimageObj, trainingmaskObj, gmms, gmmn, "training", prior_skin=prior_skin,
             prior_nonskin=prior_nonskin)

    print("TEST DATA")
    testmaskObj = imageHelper()
    testmaskObj.loadImageFromFile(os.path.join(dataPath, 'mask-test.png'))
    testimageObj = imageHelper()
    testimageObj.loadImageFromFile(os.path.join(dataPath, 'test.png'))
    classify(testimageObj, testmaskObj, gmms, gmmn, "test", prior_skin=prior_skin, prior_nonskin=prior_nonskin)

    plt.show()


if __name__ == "__main__":
    print("Python version in use: ", sys.version)
    print("\nMVND exercise - Toy example")
    print("##########-##########-##########")
    # gmmToyExample()
    print("\nMVND exercise - Skin detection")
    print("##########-##########-##########")
    gmmSkinDetection()
    print("##########-##########-##########")
