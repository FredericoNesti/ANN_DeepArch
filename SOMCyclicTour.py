#################################################################################
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict

def dist(vector): # Euclidean distance of a vector
    d = np.zeros((vector.shape[0],1))
    for i in range(vector.shape[0]):
        d[i,0] = np.sqrt(np.sum(vector[i,:]**2))
    return d

# SELF ORGANIZED MAP

class SOMCyclicTour:
    def __init__(self,
                 file='data_lab2/cities.dat',
                 fdim=(10,2),
                 weight_shape=(10,2),
                 neighbourhood_0=1,
                 step_learn=0.1,
                 n_epochs=400):
        
        # self.weight = np.random.uniform(0, 1, weight_shape)  # weights from the model
        # self.weight = 0.5 * np.ones(weight_shape)
        self.weight = np.linspace(0.,1., num=weight_shape[0]*weight_shape[1]).reshape(weight_shape)
        self.input = self.load_input(file, fdim)  # input dataset
        self.n_patterns = self.input.shape[0]  # total number of patterns
        self.n_output_patterns = self.weight.shape[0]  # number of nodes in the output grid
        self.n_neighbours = neighbourhood_0  # current number of neighbors
        self.step_learn = step_learn  # learning rate
        self.n_epochs = n_epochs  # number of epochs
        self.node2freq = np.ones(self.n_output_patterns)  # for the frequency method

    def load_input(self, filename, fdim):
        """
            return the patterns in 'filename' as a np array
        """
        with open(filename) as f:
            patterns = np.loadtxt((x.replace(';', ',') for x in f), dtype=str, delimiter=',', comments='%')
            patterns = patterns[:,[0,1]].astype(np.float)
        return patterns.reshape(fdim)

    def findRowWinnerWeight(self, cand_attr):

        distance = [0] * self.n_output_patterns
        for i in range(self.n_output_patterns):
            distance[i] = self.node2freq[i] * np.linalg.norm(self.weight[i,:] - cand_attr)
        # distance = dist(self.weight - cand_attr)
        pick_row = np.argmin(distance)
        return pick_row

    def findNeighbors(self, winner_row):
        wn_neighbors_idx = [0] * self.n_output_patterns
        curr_neigh = 0
        radius = 0
        while curr_neigh <= self.n_neighbours:
            left, right = (winner_row - radius) % self.n_output_patterns, (winner_row + radius) % self.n_output_patterns
            wn_neighbors_idx[left] = 1
            curr_neigh += 1

            if left != right and curr_neigh <= self.n_neighbours:
                wn_neighbors_idx[right] = 1
                curr_neigh += 1

            radius += 1

        return wn_neighbors_idx

    def updateWeights(self, attr, j, wn_neighbors_idx):
        for i in range(self.n_output_patterns):
            self.weight[i] += self.step_learn * wn_neighbors_idx[i] * (attr[j] - self.weight[i])

    def train(self):  # competition + neighbourhood
        
        counter_epoch = 0
        for counter_epoch in range(self.n_epochs):

            for j in range(self.n_patterns):
                # winner node
                cand_attr = self.input[j, :]
                winner_row = self.findRowWinnerWeight(cand_attr)
                self.node2freq[winner_row] += 1

                # keep winner and neighbourhood
                wn_neighbors_idx = self.findNeighbors(winner_row)

                # update winner and neighbors
                self.updateWeights(self.input, j, wn_neighbors_idx)

            # update the number of neighbors
            if counter_epoch % 50 == 0:
                self.plotResult()
                self.n_neighbours -= 1
                self.n_neighbours = max(self.n_neighbours, 0)


    def output(self, x):
        # use the SOM with this function
        return self.findRowWinnerWeight(x)

    def plotResult(self):

        labels = np.apply_along_axis(self.output, 1, self.input)

        label2city_coord = {}

        for city_coord, label in zip(self.input, labels):
            if label not in label2city_coord:
                label2city_coord[label] = str(city_coord)
            else:
                label2city_coord[label] += ' + ' + str(city_coord)

        ans = 'start'
        for label in sorted(list(label2city_coord.keys())):
            cities_coord = label2city_coord[label]
            ans += ' -> (' + str(label) + ': ' + cities_coord
        ans += '-> end'
        print(ans)

        # for city,label in zip(self.input, labels):
        #     x = [city[0], self.weight[label][0]]
        #     y = [city[1], self.weight[label][1]]
        #     plt.plot(x, y, color='m')

        plt.plot(self.weight[:, 0], self.weight[:, 1])
        plt.plot(self.weight[:, 0], self.weight[:, 1], 'ro', label='weights of model')
        plt.plot(self.input[:, 0], self.input[:, 1], 'go', label='cities points')
        plt.title('Predictions NN', fontweight="bold")
        # plt.legend(loc='upper right', borderaxespad=0.)
        plt.show()


som = SOMCyclicTour()
som.train()
som.plotResult()
