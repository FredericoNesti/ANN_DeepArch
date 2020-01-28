from Perceptron_FINAL import Perceptron
from dataset_encoder import Dataset
## Code for section 3.2.2

##### Parameters
input_dimensions = 8
ns = [3,8]  # neurons_structure
lr = 0.001  # learning rate
momentum = 0.8
tol = 3  # tolerance
eps = 500  # epochs
batchSize = 4

ds = Dataset(batchSize)

nn = Perceptron(input_dimensions=8,neurons_structure=ns,train_step=lr,train_momentum=momentum,tol=tol, max_epochs=eps)
nn.Train_NN(ds)
