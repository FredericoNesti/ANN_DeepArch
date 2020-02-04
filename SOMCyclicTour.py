#################################################################################
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


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
                 neighbourhood_0=5,
                 step_learn=0.2,
                 n_epochs=20):
        
        self.weight = np.random.uniform(0, 1, weight_shape)  # weights from the model
        self.input = self.load_input(file, fdim)  # input dataset
        self.n_patterns = self.input.shape[0]  # total number of patterns
        self.n_output_patterns = self.weight.shape[0]  # number of nodes in the output grid
        self.n_neighbours = neighbourhood_0  # current number of neighbors
        self.step_learn = step_learn  # learning rate
        self.n_epochs = n_epochs  # number of epochs

    def load_input(self, filename, fdim):
        """
            return the patterns in 'filename' as a np array
        """
        with open(filename) as f:
            patterns = np.loadtxt((x.replace(';', ',') for x in f), dtype=str, delimiter=',', comments='%')
            patterns = patterns[:,[0,1]].astype(np.float)
        return patterns.reshape(fdim)

    def findRowWinnerWeight(self, cand_attr):

        distance = dist(self.weight - cand_attr)
        pick_row = np.argmin(distance)
        return pick_row

    def findNeighbors(self, winner_row):
        wn_neighbors_idx = [0] * self.n_output_patterns
        curr_neigh = 0
        radius = 1
        while curr_neigh < self.n_neighbours:
            left, right = winner_row - radius, winner_row + radius
            if 0 <= left:
                wn_neighbors_idx[left] = 1
                curr_neigh += 1

            if right < self.n_output_patterns:
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
            
            # shuffle information for each epoch
            # train 1 attribute at a time
            shuffled_attr = np.random.permutation(self.input)
            for j in range(self.n_patterns):
                # who is winner node?
                cand_attr = shuffled_attr[j, :]
                winner_row = self.findRowWinnerWeight(cand_attr)
                
                # keep winner and neighbourhood
                wn_neighbors_idx = self.findNeighbors(winner_row)

                # update winner and neighbors
                self.updateWeights(shuffled_attr, j, wn_neighbors_idx)

            # update the number of neighbors
            self.n_neighbours -= max(1, self.n_neighbours//self.n_epochs)
   
    def output(self, x):
        # use the SOM with this function
        return self.findRowWinnerWeight(x)

    def plotResult(self):

        labels = np.apply_along_axis(self.output, 1, self.input)

        label2patter = {}

        for i,label in enumerate(labels):
            if label not in label2patter: label2patter[label] = str(i)
            else: label2patter[label] += ' + ' + str(i)

        for i in range():


        # colors = cm.rainbow(np.linspace(0, 1, self.n_output_patterns))
        # fig, ax = plt.subplots()
        #
        # for label in list(label2patter.keys()):
        #     pattern_name = label2patter[label]
        #     ax.scatter(self.weight[label][0], self.weight[label][1], color=colors[label], label=pattern_name)
        #
        # plt.title('Predictions SOM', fontweight="bold")
        # # plt.xlim(0, 125)
        # # plt.ylim(-0.005, 0.2)
        # plt.legend(ncol=3)
        # plt.show()


som = SOMCyclicTour()
som.train()
som.plotResult()
