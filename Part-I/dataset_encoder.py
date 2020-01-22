import numpy as np

class Dataset():
    """
        represents the Gauss function: f(x; y) = exp{(x2+y2)/10} − 0:5
    """
    def __init__(self,batchSize, shuffleDataset=True):
        self.batchSize = batchSize           # size of the batch during training
        self.batchPosition = 0               # used for calculate the next batch
        self.nEpochs = 0                     # actual number of epochs (used for training)

        dataset = self.generateDataset()
        self.nSamples = len(dataset[0])

        self.X = dataset[0]
        self.Y = dataset[1]

    def generateDataset(self):
        ds = np.array([np.array([-1] * 8) for i in range(9)])
        for i in range(8):
            ds[i][i] = 1

        return ds,ds

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

        return self.X[init:end], self.Y[init:end]
