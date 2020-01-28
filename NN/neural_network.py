import numpy as np


class RbfTransformation:
    def __init__(self, mean_array, sigma_array):
        self.means = mean_array
        self.sigmas = sigma_array

    def __call__(self, inp):
        ans = np.zeros(len(self.means))
        for i, (mean, sigma) in enumerate(zip(self.means, self.sigmas)):
            ans[i] = np.exp(-np.linalg.norm(inp - mean)/sigma)
        return ans


class Sigmoid:
    def func(self, x_vec):
        return 2.0 / (1.0 + np.exp(-x_vec)) - 1

    def dev_func(self, x_vec):
        return (1 + self.func(x_vec)) * (1 - self.func(x_vec)) / 2


class Linear:
    def func(self, x_vec):
        return x_vec

    def dev_func(self, x_vec):
        return 1


class Layer:
    def __init__(self, previous_layer, nodes, function=Sigmoid(), bias_weight=1, fixed_weights=False):
        self.W = np.random.rand(nodes, previous_layer) / previous_layer
        self.bias = np.random.rand(nodes) / previous_layer
        self.fixed_weights = fixed_weights
        if fixed_weights:
            self.W = np.ones(nodes, previous_layer)
            self.bias = np.zeros(nodes)
        self.function = function
        self.last_signal = -1
        self.func = function.func
        self.dev = function.dev_func
        self.bias_weight = bias_weight
        self.dw = np.zeros((nodes, previous_layer+1))

    def f_b(self, x_vec):  # return bias too
        return np.append(self.func(x_vec), [1])

    def get_wb(self):
        return np.hstack((self.W, self.bias.reshape(-1,1)))

    def update_weights(self, dw, momentum):
        if self.fixed_weights:
            return 0
        self.W += self.dw[:, :-1]*momentum + (1-momentum)*dw[:, :-1]
        self.bias += self.dw[:, -1]*momentum + (1-momentum)*dw[:, -1]
        self.dw = dw

    def feed(self, inp):
        self.last_signal = (self.W @ inp) + self.bias_weight*self.bias
        return self.last_signal, self.func(self.last_signal)


class NN:
    def __init__(self, input_num):
        # initialization of NN
        self.signal_dim = [input_num]
        self.layers = []
        self.transformations = []
        self.dw = []

    def add_transformation(self, trans):
        self.transformations.append(trans)
        o = trans(np.zeros(self.signal_dim[0]))
        self.signal_dim[0] = len(o)

    def add_layer(self, nodes, function=Sigmoid()):
        self.layers.append(Layer(self.signal_dim[-1], nodes, function))
        self.signal_dim.append(nodes)

    def feed_forward(self, inp):
        tmp = inp
        for trans in self.transformations:
            tmp = trans(tmp)
        signal_list = [tmp]
        for layer in self.layers:
            signal, tmp = layer.feed(tmp)
            signal_list.append(signal)
        return signal_list, tmp

    def back_prob(self, prediction, target, signal_list):
        dw = []
        delta = (prediction - target) * self.layers[-1].dev(signal_list[-1])
        for i in range(len(self.layers)-1, 0, -1):  # signal has length layers+1 (the input)
            dw.append(-np.outer(delta, self.layers[i-1].f_b(signal_list[i])))  # signal from previous layer
            delta = (self.layers[i].W.T @ delta)*self.layers[i-1].dev(signal_list[i])
        dw.append(-np.outer(delta, np.append(signal_list[0], [1])))
        dw.reverse()
        return dw

    def dw_batch(self, inputs, targets):
        dw = [np.zeros((layer.W.shape[0], layer.W.shape[1] + 1)) for layer in self.layers]
        for inp, trg in zip(inputs, targets):
            signals, res = self.feed_forward(inp)
            for i, dw_layer in enumerate(self.back_prob(res, trg, signals)):
                dw[i] += dw_layer/len(inputs)
        return dw

    def update_dw(self, new_dw, step, momentum):
        for i, new_dw_layer in enumerate(new_dw):
            self.layers[i].update_weights(step*new_dw_layer, momentum)

    def train_batch(self, inputs, targets, batch_size, epochs, step, momentum):
        loops = int(len(inputs)/batch_size)
        for epoch in range(0, epochs):
            for i in range(1, loops):
                inp_batch = inputs[batch_size*(i-1): batch_size*i]
                trg_batch = targets[batch_size*(i-1): batch_size*i]
                new_dw = self.dw_batch(inp_batch, trg_batch)
                self.update_dw(new_dw, step, momentum)
            inp_batch = inputs[batch_size * loops:]
            trg_batch = targets[batch_size * loops:]
            new_dw = self.dw_batch(inp_batch, trg_batch)
            self.update_dw(new_dw, step, momentum)

