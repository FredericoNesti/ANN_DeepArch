import numpy as np
import matplotlib.pyplot as plt

from numpy.random import randn
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


class Dataset():
    """
        represents the Gauss function: f(x; y) = exp{(x2+y2)/10} − 0:5
    """
    def __init__(self, n, batchSize, shuffleDataset=True):
        self.n = n                           # number os points for each class
        self.batchSize = batchSize           # size of the batch during training
        self.batchPosition = 0               # used for calculate the next batch
        self.nEpochs = 0                     # actual number of epochs (used for training)
        dataset = self.generateDataset()
        # dataset = np.insert(dataset, 0, 1, axis=1)  # add class 1 (class A = 1)
        self.nSamples = len(dataset[0])

        self.X = dataset[0]
        # self.X = np.insert(self.X, 0, 1, axis=1)
        self.Y = dataset[1]

    def plotFunctoin(self):
        x = np.arange(-3.0, 3.0, 0.1)
        y = np.arange(-3.0, 3.0, 0.1)
        X, Y = np.meshgrid(x, y)  # grid of point
        Z = np.exp(-(X**2 + Y**2)/10) - 0.5  # evaluation of the function on the grid

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=cm.RdBu, linewidth=0, antialiased=False)

        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()

    def generateDataset(self):
        def f(array):
            return np.exp(-(array[0]**2 + array[1]**2)/10) - 0.5

        x = ((np.random.rand(self.n, 2))-0.5)*6
        z = np.apply_along_axis(f, 1, x)

        return x, z

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

# Exemple of use:

ds = Dataset(10, 3)

while ds.nEpochs < 5:
    print(ds.nextBatch())

