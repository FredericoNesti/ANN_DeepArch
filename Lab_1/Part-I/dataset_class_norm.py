import numpy as np
import matplotlib.pyplot as plt

from numpy.random import randn

class Dataset():
    """
        represents a multivariate normal distribution
    """
    def __init__(self, n, mA, sigmaA, mB, sigmaB, batchSize, shuffleDataset=True):
        self.n = n                           # number os points for each class
        self.dim = len(mA)                   # dimention of normal distribution
        self.mA = mA                         # mean of normal A
        self.sigmaA = sigmaA                 # standard deviation A
        self.covA = np.eye(self.dim)*sigmaA  # covariance of A
        self.mB = mB                         # mean of normal B
        self.sigmaB = sigmaB                 # standard deviation B
        self.covB = np.eye(self.dim)*sigmaB  # covariance of B
        self.batchSize = batchSize           # size of the batch during training
        self.batchPosition = 0               # used for calculate the next batch
        self.nEpochs = 0                     # actual number of epochs (used for training)

        self.classA = np.random.multivariate_normal(self.mA, self.covA, self.n).T  # generate normal
        self.classA = np.insert(self.classA, 0, 1, axis=0)  # add class 1 (class A = 1)

        self.classB = np.random.multivariate_normal(self.mB, self.covB, self.n).T  # generate normal
        self.classB = np.insert(self.classB, 0, [-1], axis=0)  # add class -1 (class B = -1)

        dataset = np.concatenate((self.classA, self.classB), axis=1)
        self.nSamples = len(dataset[0])

        if shuffleDataset:
            np.random.shuffle(dataset.T)

        self.X = dataset[1:, :].T
        self.Y = dataset[0, :].T


    def printDataset(self, weights):
        """
        :return: it prints the dataset with the boarder separation of weights
        """
        x = np.arange(min(self.X[:, 0]), max(self.X[:, 1]), 0.001)
        y = - (weights[0][1] / weights[0][2]) * x - weights[0][0]/weights[0][2]
        plt.plot(x, y, label="separation line")
        plt.plot(self.classA[1, :], self.classA[2,:], 'ro', label="class A")
        plt.plot(self.classB[1, :], self.classB[2,:], 'go', label="class B")
        plt.title('Dataset plot', fontweight="bold")
        plt.legend(bbox_to_anchor=(0.05, .95), loc='upper left', borderaxespad=0.)
        plt.show()


    def nextBatch(self):
        """
        :return: the next batch of the dataset
        """
        if self.batchPosition == self.nSamples:
            self.batchPosition = 0
            self.nEpochs += 1

        init = self.batchPosition
        end = min(self.nSamples, self.batchPosition + self.batchSize)
        self.batchPosition = end

        return self.X[init:end, :], self.Y[init:end]

# example of use
# Parameters

n = 100
mA = [ 1.0, 0.5]
sigmaA = 0.5
mB = [-3, -1.5]
sigmaB = 0.5
batchSize = 20

lr = 0.001
n_epochs = 30

dataset = Dataset(n, mA, sigmaA, mB, sigmaB, batchSize)

while dataset.nEpochs < n_epochs:
    batch_patterns, batch_target = dataset.nextBatch()
    #print(batch_patterns)
    #print('\n')
    # model.updateWeightsDeltaRule(batch_patterns, batch_target)
