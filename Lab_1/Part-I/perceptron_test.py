import numpy as np
from dataset_class_norm import Dataset
from perceptron_batch import Perceptron
n = 100
batchSize = 10 #batch sizes 2 and 1 gives problem
mA = [ 1.0, 0.5]
sigmaA = 0.5**2
mB = [-3, -1.5]
sigmaB = 0.5**2
dataset = Dataset(n, mA, sigmaA, mB, sigmaB, batchSize)
perc = Perceptron(3,1)
perc.batch_method(dataset)
print(perc.weight)
print('Epochs: ' + str(dataset.nEpochs))
dataset.printDataset(perc.weight)