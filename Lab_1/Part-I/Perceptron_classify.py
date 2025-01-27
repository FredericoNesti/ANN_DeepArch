# In[0]:
#### Libraries
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
#from dataset_reg import Dataset
from dataset_class_norm import Dataset
#from dataset_class_norm import *

# In[1]:
#### Generate databases

#x =  np.arange(0,10,0.5).reshape(-1,1)
#y = np.random.normal(0,1,x.shape[0])

np.random.seed(6)

n = 100
batchSize = 1 #batch sizes 2 and 1 gives problem
mA = [ 1.0, 0.5]
sigmaA = 0.5**2
mB = [-3, -1.5]
sigmaB = 0.5**2


dataset = Dataset(n, mA, sigmaA, mB, sigmaB, batchSize)

# In[2]:
#### ANNs

def Activation_Func(X_vec):
    #Sigmoid function
    #return X_vec**2
    return 2.0/(1.0 + np.exp(-X_vec)) -1

def d_Activation_Func(X_vec):
    # Derivative of the Activation Function
    #return 2*X_vec
    return (1.0+Activation_Func(X_vec))*(1.0-Activation_Func(X_vec))/2.0

def Activation_Output(X_vec):
    #Linear activation
    #return X_vec
    return 2.0/(1.0 + np.exp(-X_vec)) -1

def d_Activation_Output(X_vec):
    # Derivative of the Activation Function for the last layer
    #return np.ones((X_vec.shape))
    return (1.0+Activation_Output(X_vec))*(1.0-Activation_Output(X_vec))/2.0

class Perceptron():
    #without Bias
    def __init__(self,input_dimensions,neurons_structure,train_step,train_momentum,tol, max_epochs=500):
        
        self.momentum = train_momentum
        self.tol = tol
        self.train_step = train_step
        self.neurons_structure = neurons_structure
        self.no_inputs = input_dimensions
        self.no_layers = len(neurons_structure)
        self.Init_All_Weights()
        self.no_epochs = 0
        self.early_stop=0
        self.max_epochs = max_epochs
        #self.flag_bias = bias #####
        
    def Init_All_Weights(self):
        self.mean_of_signals = []
        self.all_weights = []
        
        self.all_bias = [] #####
        
        i_prevlayer = 0
        for pos, i in enumerate(self.neurons_structure):
            if pos == 0:
                # init weights (Normal 0 1)
                self.all_weights.append(np.random.normal(0,1,(self.no_inputs)*i).reshape((self.no_inputs),i))
                self.all_bias.append(np.random.normal(0,1,i).reshape(1,i)) #####
            else:
                # init weights (Normal 0 1)
                self.all_weights.append(np.random.normal(0,1,(i_prevlayer)*i).reshape((i_prevlayer),i))
                self.all_bias.append(np.random.normal(0,1,i).reshape(1,i)) #####
            
            i_prevlayer = i
        
        # initialize old_updates for training w/ momentum
        self.old_updates = [np.zeros((self.all_weights[i].shape)) for i in range(self.no_layers)]
        self.old_updates_forbias = [np.zeros((self.all_bias[i].shape)) for i in range(self.no_layers)] ######
        
    def Train_NN(self, dataset):
        # dataset = inputs,targets
        # the first delta in the recorded list is the last delta in the network
        self.last_loss = 10.0 #arbitrary
        self.old_loss = 0.0 #arbitrary
        self.errorlist = []
        totallos = 0
        while (self.early_stop <=dataset.nSamples/dataset.batchSize or dataset.nEpochs < 1) and self.no_epochs < self.max_epochs:
            batch_input, batch_target = dataset.nextBatch()

            self.no_epochs += 1
            self.old_loss = self.last_loss 
            #print("batch input")
            #print(batch_input)
            #print("batch target")
            #print(batch_target)
            self.Forward_step(batch_input)
            self.batch_num = dataset.batchSize
            
            self.Backprop_train(batch_input, batch_target)
            if self.last_loss==0:
                self.early_stop+=1
            else:
                self.early_stop=0
            print('Epoch: ', self.no_epochs)
            #print('Weights')
            #print(self.all_weights)
            #print('Signals')
            #print(self.all_signals)
            #print('Updates')
            #print(self.all_updates)
            #print('#####################')
            totallos+=self.last_loss
            if dataset.batchPosition == dataset.nSamples:
                self.errorlist.append(totallos)
                totallos=0
            
        print('Epoch: ', self.no_epochs)
        
    def Forward_step(self,inputs):
        # inputs are from batch
        #Propagating message (linear combinations)
        self.batch_signals = []
        for j in range(inputs.shape[0]):
            all_signals = []
            for i in range(self.no_layers):
                if i == 0:
                    aux2 = self.all_bias[i].T #@ np.ones((inputs[j].shape))
                    aux1 = (self.all_weights[i].T @ inputs[j]).reshape(-1,1)
                    
                    #print(aux2.shape)
                    #print(aux1.shape)
                    
                    #print('check here')
                    #print((aux1 + aux2).shape)
                    
                    all_signals.append( aux1 + aux2 )
                    del(aux1,aux2)
                    
                else:
                    aux3 = self.all_bias[i].T #@ np.ones((Activation_Func(all_signals[i-1]).shape))
                    aux4 = self.all_weights[i].T @ Activation_Func(all_signals[i-1])
                    
                    #print(aux3.shape)
                    #print(aux4.shape)
                    
                    all_signals.append( aux3 + aux4 )
                    del(aux3,aux4)
                    
            self.batch_signals.append(all_signals)
            
    def Output(self,one_input):
        # for activating just the last signal = network output
        self.Forward_step(one_input.reshape(1,-1))
        return Activation_Output(self.batch_signals[0][-1])
    
    def Backprop_train(self,inputs,targets):
        # everything is in an opposite way
        self.last_loss = 0
        sum_updates = [np.zeros((self.all_weights[i].shape)) for i in range(self.no_layers-1,-1,-1)]
        sum_updates_bias = [np.zeros((self.all_bias[i].shape)) for i in range(self.no_layers-1,-1,-1)]
        for j,signal_of_each_input in enumerate(self.batch_signals):
            
            # activate signal of just the last layer
            predictions = Activation_Output(signal_of_each_input[-1])
            d_o_ = d_Activation_Output(signal_of_each_input[-1])
            #print(predictions)
            #print(targets[j])
            self.last_loss += np.sum(np.abs(((predictions>=0)*2-1)-targets[j])/2)
            tmp_deltas = ((predictions-targets[j])*d_o_).reshape(-1,1)
            
            tmp_deltas_bias = ((predictions-targets[j])*d_o_).reshape(-1,1)
            
            self.all_updates = []
            self.all_updates_forbias = []
            for i in range(self.no_layers-1,0,-1):
                
                self.all_updates_forbias.append(self.train_step*(self.momentum*self.old_updates_forbias[i]-(1-self.momentum)*(tmp_deltas_bias ).T))    
                self.all_updates.append(self.train_step*(self.momentum*self.old_updates[i]-(1-self.momentum)*(tmp_deltas @ Activation_Func(signal_of_each_input[i-1]).reshape(1,-1)).T))    
                
                tmp_deltas_bias = ((self.all_bias[i] @ tmp_deltas_bias) * (d_Activation_Func(signal_of_each_input[i-1])).reshape(-1,1)).reshape(-1,1)
                tmp_deltas = ((self.all_weights[i] @ tmp_deltas) * (d_Activation_Func(signal_of_each_input[i-1])).reshape(-1,1)).reshape(-1,1)
            
            self.all_updates_forbias.append(self.train_step*(self.momentum*self.old_updates_forbias[0]-(1-self.momentum)*(tmp_deltas_bias ).T)) 
            self.all_updates.append(self.train_step*(self.momentum*self.old_updates[0]-(1-self.momentum)*(tmp_deltas @ inputs[j].reshape(1,-1)).T)) 
            
            
            for i in range(len(self.all_updates)):
                sum_updates[i]+= self.all_updates[i]/self.batch_num
                sum_updates_bias[i]+= self.all_updates_forbias[i]/self.batch_num
        
        self.old_updates = sum_updates.copy()
        # we want to write it backwards (see Backpropagation function)
        self.old_updates.reverse()
        
        for i in range(self.no_layers):
            # weights is written from left to right
            # updates are written from right to left (check minus signal)
            self.all_weights[i] += sum_updates[-(i+1)]
            self.all_bias[i] += sum_updates_bias[-(i+1)]
   
# In[3]:

def main(ds,neuron_topol,step,mom,tol,eps):
    
    np.random.seed(6)
    
    ### TRAIN
    Net = Perceptron(ds.X.shape[1],neuron_topol,step,mom,tol,eps)
    Net.Train_NN(ds)
    #W = Net.all_weights
    #S = Net.all_signals
    #print('Pesos')
    #print(W)
    #print('')
    #print('Sinais')
    #print(S)
 
    ### PLOTTING
    
    w = Net.all_weights[0].reshape(-1)
    b = Net.all_bias[0][0]
    sse = 0
    #w = np.flip(w)
    w = np.insert(w,0,b,axis=0)
    print("Epochs used")
    print(ds.nEpochs)
    ds.printDataset([w])
    plt.show()
    plt.figure()
    Net.errorlist.append(0)
    plt.plot(np.array(Net.errorlist))
    plt.xlabel('Epochs')
    plt.ylabel('Misclassified points')
    plt.show()
    print(np.array(Net.errorlist))
    print('Error: ', sse)
    print('Sqrt Error: ', np.sqrt(sse))

# In[4]:

if __name__ == "__main__":
    import time
    start_time = time.time()
    main(dataset, [1], 0.01, 0.0, 1e-06,100000)
    print("--- %s seconds ---" % (time.time() - start_time))
    




# %%
