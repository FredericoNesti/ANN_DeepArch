#################################################################################

import numpy as np
import os

def dist(vector): # Euclidean distance of a vector
    d = np.zeros((vector.shape[0],1))
    for i in range(vector.shape[0]):
        d[i,0] = np.sqrt(np.sum(vector[i,:]**2))
    return d

# SELF ORGANIZED MAP
class SOM:
    
    def __init__(self,
                 file,
                 fpath = 'C:/Users/frede/Desktop/Academic/KTH/ANN/Lab_2',
                 fdim=(32,84),
                 ftype=int,
                 fdelim = ',',
                 weight_shape=(100,84),
                 neighbourhood_0 = [10,10],
                 step_learn = 0.2,
                 n_epochs = 20,
                 neighbourhood_type = 'linear'): # either linear or circular
        
        self.weight = np.random.uniform(0,1,weight_shape) #random initialization
        self.n_animals = fdim[1]
        os.chdir(fpath) #set file path directory
        
        self.input = np.loadtxt(file,dtype=ftype,delimiter=fdelim).reshape(fdim) # load file -> "props" matrix
        
        self.neighbours = neighbourhood_0
        self.step_learn = step_learn
        self.n_epochs = n_epochs
        self.neib_type = neighbourhood_type
    
    
    def train(self): # competition + neighbourhood
        
        counter_epoch = 0
        while counter_epoch < self.n_epochs:
            
            # shuffle information for each epoch
            # train 1 attribute at a time
            self.shuffled_attr = np.random.permutation(self.input)
            
            for j in range(self.n_animals):
                
                # who is winner node?
                picked_cand = self.shuffled_attr[j,:] # shuffled "props" matrix
                distance = dist(self.weight - picked_cand)
                pick_row = np.argmin(distance)
                
                # keep winner and neighbourhood
                wn_indexes =  [(pick_row+i)%self.n_animals for i in range(self.neighbours[1])] # winner and neighbour indexes
                for i in range(self.neighbours[0]):
                    wn_indexes.append((pick_row-i)%self.n_animals)
                
                print(wn_indexes)
                
                wn_indexes = np.sort(wn_indexes)
                
                # correct updates according to type of neighbourhood
                if self.neib_type == 'linear':
                    wn_fix =  wn_indexes[:(pick_row-self.neighbours[0])]
                elif self.neib_type == 'circular':
                    wn_fix =  wn_indexes[:]
                else:
                    print('Enter valid format!')
                
                # who will be updated? winner and neighbourhood
                update_indicator = np.zeros((self.n_animals))
                update_indicator[wn_fix] = np.ones(sum(self.neighbours))
                
                # ok, found winner -> update weights
                self.weight += update_indicator*self.step_learn*(distance - self.weight)
                
            # start a new epoch after all animals
            counter_epoch += 1     
   
    def fwd(self):
        # use the SOM with this function
        self.output = self.weight @ dist(self.input.T)
        return self.output

