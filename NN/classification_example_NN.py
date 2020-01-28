import numpy as np
import matplotlib.pyplot as plt
from neural_network import *
from dataset_class_norm import Dataset
# creating the Dataset
mA = [1.0, 0.5]
sigmaA = 0.5**2
mB = [-3, -1.5]
sigmaB = 0.5**2
data = Dataset(100, mA, sigmaA, mB, sigmaB, 20)
# creating the network
nn = NN(2)  # argument the number of inputs
nn.add_layer(1)  # adding layer, we can also choose the activation function
# training the network
nn.train_batch(data.X, data.Y, 20, 1, 0.05, 0.6)  # here we train setting parameters (batch_size, epoch, step, momentum)
# --------------
# check accuracy
# --------------
correct = 0
for i in range(len(data.X)):
    signals, out = nn.feed_forward(data.X[i])
    out = out[0]
    print(out, data.Y[i])
    if out*data.Y[i] > 0:
        correct+=1
print("accuracy: " + str(correct/len(data.X)))
# --------------
# plotting
# --------------
w = nn.layers[0].W[0]
b = nn.layers[0].bias[0]
print(w, b)
w = np.insert(w, 0, b, axis=0)
print(w)
data.printDataset([w])

