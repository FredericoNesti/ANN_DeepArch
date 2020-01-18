import numpy as np

class Perceptron():
    def __init__(self, s_input, s_output, lr):
        self.s_input = s_input
        self.s_output = s_output
        self.lr = lr                # learning rate
        self.w = np.random.normal(s_input*s_output, size=(s_output, s_input)) # matrix of weights

    def updateWeightsDeltaRule(self, x, y_true):
        self.w = self.w - self.lr * (np.matmul((np.matmul(self.w, x) - y_true), x.T))
