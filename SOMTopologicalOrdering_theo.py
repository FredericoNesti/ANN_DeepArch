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


class SOMTopologicalOrdering:
    def __init__(self,
                 file='NN/Lab_2_data/Datasets/animals.dat',
                 animals_name='NN/Lab_2_data/Datasets/animalnames.txt',
                 fdim=(32,84),
                 ftype=int,
                 fdelim = ',',
                 weight_shape=(100,84),
                 neighbourhood_0 = 20,
                 step_learn = 0.2,
                 n_epochs = 20,
                 neighbourhood_type='linear'): # either linear or circular
        
        self.weight = np.random.uniform(0,1,weight_shape) #random initialization

        self.input = np.loadtxt(file,dtype=ftype,delimiter=fdelim).reshape(fdim) # load file -> "props" matrix
        self.animals_name = np.loadtxt(animals_name, dtype=str)
        self.n_patterns = self.input.shape[0]
        self.n_output_patterns = self.weight.shape[0]

        self.n_neighbours = neighbourhood_0
        self.step_learn = step_learn
        self.n_epochs = n_epochs
        self.neib_type = neighbourhood_type

    def findRowWinnerWeight(self, candidate):

        distance = dist(self.weight - candidate)
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

    def train(self): # competition + neighbourhood
        
        counter_epoch = 0
        while counter_epoch < self.n_epochs:
            
            # shuffle information for each epoch
            # train 1 attribute at a time
            shuffled_attr = np.random.permutation(self.input)
            for j in range(self.n_patterns):
                
                # who is winner node?
                picked_cand = shuffled_attr[j, :]
                winner_row = self.findRowWinnerWeight(picked_cand)
                
                # keep winner and neighbourhood
                wn_neighbors_idx = self.findNeighbors(winner_row)

                # update winner and neighbors
                self.updateWeights(shuffled_attr, j, wn_neighbors_idx)
                
            # start a new epoch after all animals
            counter_epoch += 1

            # update the number of neighbors
            self.n_neighbours -= max(1, self.n_neighbours//self.n_epochs)
   
    def output(self, x):
        # use the SOM with this function
        return self.findRowWinnerWeight(x)

    def plotResult(self):

        labels = np.apply_along_axis(self.output, 1, self.input)

        label2animals = {}

        for label, ani_name in zip(labels, self.animals_name):
            if label not in label2animals: label2animals[label] = ani_name.replace('\'', '')
            else: label2animals[label] += ' + ' + ani_name.replace('\'', '')

        colors = cm.rainbow(np.linspace(0, 1, self.n_output_patterns))
        fig, ax = plt.subplots()

        for label in sorted(list(label2animals.keys())):
            ani_name = label2animals[label]
            ax.scatter(label, [0], color=colors[label], label=ani_name)

        plt.title('Predictions SOM', fontweight="bold")
        plt.xlim(0, 125)
        plt.ylim(-0.005, 0.2)
        plt.legend(loc='upper right', borderaxespad=0., ncol=3)
        plt.show()


som = SOMTopologicalOrdering(neighbourhood_type='linear')
som.train()
som.plotResult()
