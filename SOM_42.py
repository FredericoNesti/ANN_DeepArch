
import numpy as np
import os

def dist(vector): # Euclidean distance of a vector
    d = np.zeros((vector.shape[0],1))
    for i in range(vector.shape[0]):
        d[i,0] = np.sqrt(np.sum(vector[i,:]**2))
    return d

def remove_duplicate(duplicate):
    final_list = []
    for num in duplicate:
        if num not in final_list:
            final_list.append(num)
    return final_list



# SELF ORGANIZED MAP
class SOM:
    
    def __init__(self,
                 file,
                 fpath = 'C:/Users/frede/Desktop/Academic/KTH/ANN/Lab_2',
                 fdim=(10,2),
                 ftype=int,
                 fdelim = ',',
                 weight_shape=(10,2),
                 neighbourhood_0 = [3,3], #border excluded
                 step_learn = 0.2,
                 n_epochs = 50): # either linear or circular
        
        self.file = file
        self.weight = np.random.uniform(0,1,weight_shape) #random initialization
        self.n_animals = fdim[0]
        os.chdir(fpath) #set file path directory
        
        #self.input = np.loadtxt(self.file,dtype=ftype,delimiter=fdelim).reshape(fdim) # load file -> "props" matrix
        self.input = np.loadtxt('cities2.txt',dtype=float,delimiter=',').reshape(fdim) # load file -> "props" matrix
        
        self.neighbours = neighbourhood_0
        self.step_learn = step_learn
        self.n_epochs = n_epochs
        self.shuffled_attr = np.random.permutation(self.input)
    
    def train(self): # competition + neighbourhood
        
        counter_epoch = 1
        while counter_epoch <= self.n_epochs:
            
            print('Training: epoch number',counter_epoch,' of ',self.n_epochs)
            
            # shuffle information for each epoch
            # train 1 attribute at a time
            self.shuffled_attr = np.random.permutation(self.input)
            
            for j in range(self.n_animals):
                
                # who is winner node?
                picked_cand = self.shuffled_attr[j,:] # shuffled "props" matrix
                distance = dist(self.weight - picked_cand)
                #print(distance)
                pick_row = np.argmin(distance) # for the winner
                #print(pick_row)
                
                # keep winner and neighbourhood (circular case)
                wn_indexes = [pick_row] # winner and neighbour indexes
                for i in range(self.neighbours[1]):
                    wn_indexes.append((pick_row+i)%self.n_animals)
                for i in range(self.neighbours[0]):
                    wn_indexes.append((pick_row-i)%self.n_animals)
                wn_indexes = remove_duplicate(np.sort(wn_indexes).tolist())
                #wn_indexes = np.sort(wn_indexes)
                
                #print(wn_indexes)

                # who will be updated? winner and neighbourhood
                update_indicator = np.zeros((self.n_animals,1))
                update_indicator[wn_indexes,0] = np.ones(sum(self.neighbours)-1).tolist()
                
                # ok, found winner -> update weights
                
                '''
                print(self.weight.shape)
                print(distance.shape)
                print(update_indicator.shape)
                '''
                
                self.weight += update_indicator*self.step_learn*(distance - self.weight)
                
            # start a new epoch after all animals
            # reset neighbours according to epoch
            counter_epoch += 1     
            
            if self.n_epochs%counter_epoch == 4: # in the thrid part of the goal epoch we reuce the neighbours
                self.neighbours[0] = max(0,self.neighbours[0] - 1)
                self.neighbours[1] = max(0,self.neighbours[1] - 1)
                
    def use(self,x_in):
        # for using SOM pick just the winner
        
        winner_index = np.argmin(dist(self.weight - x_in))
        winner_indicator = np.zeros((self.n_animals,1))
        winner_indicator[winner_index,0] = 1
        
        return winner_indicator
            


    
