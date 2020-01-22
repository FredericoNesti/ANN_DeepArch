# In[0]:
#### Libraries
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
#from dataset_reg import Dataset
from dataset_reg_func_approx import Dataset
#from dataset_class_norm import *

# In[1]:
#### Generate databases

#x =  np.arange(0,10,0.5).reshape(-1,1)
#y = np.random.normal(0,1,x.shape[0])

np.random.seed(6)

n = 100
batchSize = 10 #batch sizes 2 and 1 gives problem

dataset = Dataset(n,batchSize)

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
    return X_vec
    #return 2.0/(1.0 + np.exp(-X_vec)) -1

def d_Activation_Output(X_vec):
    # Derivative of the Activation Function for the last layer
    return np.ones((X_vec.shape))
    #return (1.0+Activation_Output(X_vec))*(1.0-Activation_Output(X_vec))/2.0

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
        
        while np.abs(self.last_loss - self.old_loss) >= self.tol and self.no_epochs < self.max_epochs:
            batch_input, batch_target = dataset.nextBatch()
    
            self.no_epochs += 1
            self.old_loss = self.last_loss 
            
            self.Forward_step(batch_input)
            self.batch_num = dataset.batchSize
                
            self.Backprop_train(batch_input, batch_target)
            
            print('Epoch: ', self.no_epochs)
            #print('Weights')
            #print(self.all_weights)
            #print('Signals')
            #print(self.all_signals)
            #print('Updates')
            #print(self.all_updates)
            #print('#####################')
            
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
            self.last_loss += np.sum(predictions-targets[j])
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

def main(ds,neuron_topol,step,mom,eps):
    
    np.random.seed(6)
    
    ### TRAIN
    Net = Perceptron(ds.X.shape[1],neuron_topol,step,mom,eps)
    Net.Train_NN(ds)
    #W = Net.all_weights
    #S = Net.all_signals
    #print('Pesos')
    #print(W)
    #print('')
    #print('Sinais')
    #print(S)
 
    ### PLOTTING
    ds.plotFunctoin()
    plt.show()
    x = np.arange(-3.0, 3.0, 0.1)
    y = np.arange(-3.0, 3.0, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    sse = 0
    for index,(x,y) in enumerate(zip(X,Y)):
        for index2,(x2,y2) in enumerate(zip(x,y)):
            Z[index,index2] = Net.Output(np.array([x2,y2]))
            sse+=(Z[index,index2] - np.exp(-(x2**2 + y2**2)/10) - 0.5)**2

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                            cmap=cm.RdBu, linewidth=0, antialiased=False)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
    
    print('Error: ', sse)
    print('Sqrt Error: ', np.sqrt(sse))

# In[4]:

if __name__ == "__main__":
    import time
    start_time = time.time()
    main(dataset, (25,1), 0.3, 0.0, 1e-06)
    print("--- %s seconds ---" % (time.time() - start_time))
    


