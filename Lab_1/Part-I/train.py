from dataset import Dataset
from perceptron import Perceptron

##############################################
# Parameters

n = 100
mA = [ 1.0, 0.5]
sigmaA = 0.5
mB = [-3, -1.5]
sigmaB = 0.5
batchSize = 20

lr = 0.001
n_epochs = 30

##############################################

dataset = Dataset(n, mA, sigmaA, mB, sigmaB, batchSize)

model = Perceptron(3, 1, lr)

while dataset.nEpochs < n_epochs:
    batch_patterns, batch_target = dataset.nextBatch()
    model.updateWeightsDeltaRule(batch_patterns, batch_target)

dataset.printDataset(model.w)

