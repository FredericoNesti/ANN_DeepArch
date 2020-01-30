import numpy as np


class RbfTransformation:
    def __init__(self, mean_array, sigma_array):
        self.means = mean_array
        self.sigmas = sigma_array

    def __call__(self, inp):
        ans = np.zeros(len(self.means))
        for i, (mean, sigma) in enumerate(zip(self.means, self.sigmas)):
            ans[i] = np.exp(-(np.inner(inp - mean, inp - mean))/sigma/2)
        return ans

    def train_mu(self, data, step, epochs):
        for epoch in range(epochs):
            for sample in np.random.choice(data, len(data)):
                minn = np.inf
                minn_i = -1
                for i in range(len(self.means)):
                    distance = np.linalg.norm(sample-self.means[i])
                    if minn > distance:
                        minn = distance
                        minn_i = i
                self.means[minn_i] += step*(sample-self.means[minn_i])


class Sigmoid:
    def __init__(self, slope=1):
        self.slope = slope

    def func(self, x_vec):
        return 2.0 / (1.0 + np.exp(-self.slope*x_vec)) - 1

    def dev_func(self, x_vec):
        return ((1 + self.func(x_vec)) * (1 - self.func(x_vec)) / 2)*self.slope


class Relu:
    def func(self, x_vec):
        return (x_vec > 0)*1*x_vec

    def dev_func(self, x_vec):
        return (x_vec > 0)*1


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
        self.W += self.dw[:, :-1]*momentum + dw[:, :-1]  # the lab 1 says (1-momentum)* but i believe is incorrect
        self.bias += self.dw[:, -1]*momentum + dw[:, -1]
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

    def add_layer(self, nodes, function=Sigmoid(), **args):
        self.layers.append(Layer(self.signal_dim[-1], nodes, function, **args))
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

    def train_batch(self, inputs, targets, batch_size, epochs, step, momentum, learning_curve=False, test_set=None, test_targets=None):
        training_error = []
        test_error = []
        loops = int(len(inputs)/batch_size)
        for epoch in range(0, epochs):
            prev = 0
            for i in range(0, loops):
                inp_batch = inputs[prev: batch_size*(i+1)]
                trg_batch = targets[prev: batch_size*(i+1)]
                new_dw = self.dw_batch(inp_batch, trg_batch)
                self.update_dw(new_dw, step, momentum)
                prev = batch_size*(i+1)
            inp_batch = inputs[batch_size * loops:]
            trg_batch = targets[batch_size * loops:]
            new_dw = self.dw_batch(inp_batch, trg_batch)
            self.update_dw(new_dw, step, momentum)
            if learning_curve:
                error = 0
                for s, t in zip(inputs, targets):
                    error += np.abs(self.feed_forward(s)[1][0]-t)
                error = error/len(inputs)
                training_error.append(error)
            error = 0
            if test_set is not None:
                for s, t in zip(test_set, test_targets):
                    error += np.abs(self.feed_forward(s)[1][0]-t)
                error = error / len(test_set)
                test_error.append(error)
        return training_error, test_error


