import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from neural_network import *
from dataset_reg_func_approx import Dataset
# creating the Dataset
n = 100
batchSize = 10
np.random.seed(6)
data = Dataset(n, batchSize)
# creating the network
nn = NN(2)  # argument the number of inputs
# adding layer, we can also choose the activation function default is sigmoid,  you can also disable the bias
nn.add_layer(15)
nn.add_layer(1)  # adding layer, we can also choose the activation function
# training the network
nn.train_batch(data.X, data.Y, 10, 150, 1, 0)  # here we train setting parameters (batch_size, epoch, step, momentum)
# --------------
# plotting
# --------------
x = np.arange(-3.0, 3.0, 0.1)
y = np.arange(-3.0, 3.0, 0.1)
X, Y = np.meshgrid(x, y)
Z = np.zeros(X.shape)
sse = 0
for index, (x, y) in enumerate(zip(X, Y)):
    for index2, (x2, y2) in enumerate(zip(x, y)):
        Z[index, index2] = nn.feed_forward(np.array([x2, y2]))[1]
        sse += (Z[index, index2] - np.exp(-(x2**2 + y2**2)/10) - 0.5)**2

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                        cmap=cm.RdBu, linewidth=0, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
print("SSE: " + str(sse))

