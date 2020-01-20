# In[0]:
#### Libraries
import numpy as np
import matplotlib.pyplot as plt
from dataset_reg import Dataset #import instead dataset_reg_func_approx

# In[1]:
#### Generate databases

#x =  np.arange(0,10,0.5).reshape(-1,1)
#y = np.random.normal(0,1,x.shape[0])

n = 100
batchSize = 20

dataset = Dataset(n,batchSize)

# In[2]:
#### ANNs

def Activation_Func(X_vec):
    #Sigmoid function
    return 1.0/(1.0 + np.exp(-X_vec)) 

def d_Activation_Func(X_vec):
    # Derivative of the Activation Function
    return (1.0+Activation_Func(X_vec))*(1.0-Activation_Func(X_vec))/2.0

def Activation_Output(X_vec):
    #Linear activation
    return X_vec

def d_Activation_Output(X_vec):
    # Derivative of the Activation Function for the last layer
    return np.ones((X_vec.shape))

class Perceptron():
    #without Bias
    
    '''
    WARNING !!!!
    
    This perceptron is without bias implementation
    Remember: bias is layer by layer
    '''
       
    def __init__(self,input_dimensions,neurons_structure,train_step,train_momentum,tol, max_epochs=1000):
        
        self.momentum = train_momentum
        self.tol = tol
        self.train_step = train_step
        self.neurons_structure = neurons_structure
        self.no_inputs = input_dimensions
        self.no_layers = len(neurons_structure)
        self.Init_All_Weights()
        self.no_epochs = 0
        self.max_epochs = max_epochs
        
    def Init_All_Weights(self):
        
        self.all_weights = []
        i_prevlayer = 0
        for pos, i in enumerate(self.neurons_structure):
            if pos == 0:
                # init weights (Normal 0 1)
                aux = np.random.normal(0,1,(self.no_inputs)*i).reshape((self.no_inputs),i)
            else:
                # init weights (Normal 0 1)
                aux = np.random.normal(0,1,(i_prevlayer)*i).reshape((i_prevlayer),i)
            i_prevlayer = i
            self.all_weights.append(aux)
        
        # initialize old_updates for training w/ momentum
        self.old_updates = [np.zeros((self.all_weights[i].shape)) for i in range(self.no_layers)]
        
    def Train_NN(self, dataset):
        # inputs,targets
        # the first delta in the recorded list is the last delta in the network
        self.last_loss = 10.0 #arbitrary
        self.old_loss = 0.0 #arbitrary
        
        while np.abs(self.last_loss - self.old_loss) >= self.tol and self.no_epochs < self.max_epochs:
            batch_input, batch_target = dataset.nextBatch()
    
            self.no_epochs += 1
            self.old_loss = self.last_loss 
            o_ = []
            
            for i in range(dataset.batchSize):
                # the input must be read tronsposed because of package dataset (diff config from here)
                #self.Forward_step(batch_input[i,:])
                
                print('first')
                print(self.Output(batch_input[i,:]))
                print(Activation_Output(self.Output(batch_input[i,:])))
                print('second')
                
                o_.append( Activation_Output(self.Output(batch_input[i,:])).tolist()[0] )
                
                print(o_)
            
            self.Backprop_train(batch_input, np.array(o_), batch_target)            
            self.last_loss = self.all_deltas[0]
            
        print('Epoch: ', self.no_epochs)
        
    def Forward_step(self,inputs):
        #Propagating message (linear combinations)
        self.all_signals = []
        i_prevlayer = 0
        for i in range(self.no_layers):
            if i == 0:
                self.all_signals.append( self.all_weights[i].T @ inputs )
            else:
                self.all_signals.append( self.all_weights[i].T @ Activation_Func(self.all_signals[i_prevlayer]) )
            i_prevlayer = i
            
    def Output(self,inputs):
        # for activating just the last signal = network output
        self.Forward_step(inputs)
        return Activation_Output(self.all_signals[-1])
        
    def Backprop_train(self,inputs,predictions,targets):
        # everything is in an opposite way
        d_o_ = d_Activation_Output(predictions)
        
        aux = (predictions-targets)*d_o_
        print((predictions-targets).shape)
        print(d_o_.shape)
        print(aux.shape)
        print('checkpoint')
        
        self.all_deltas = [sum((predictions-targets)*d_o_)]
        
        print(self.all_deltas)
        
        self.all_updates = []
        
        for i in range(self.no_layers-1,0,-1):
            
            print('here')
            print(np.array(self.all_deltas[-1]))
            print(self.all_weights[i].T[0])
            print(d_Activation_Func(self.all_signals[i-1]))
            print('here2')
            
            self.all_deltas.append(( np.array(self.all_deltas[-1]).reshape(-1,self.all_weights[i].T[0].shape[0]) @ self.all_weights[i].T[0]) * (d_Activation_Func(self.all_signals[i-1])) )
            
            new_info = np.tile(self.all_deltas[-1] @ self.all_signals[i],self.old_updates[i].shape)
            
            self.all_updates.append( self.train_step * ( self.momentum * self.old_updates[i] - (1-self.momentum) * new_info ))
            #self.all_updates.append(self.train_step*(self.momentum*self.old_updates[i]-(1-self.momentum)*(self.all_deltas[-1] @ self.all_signals[i])))
            
        new_info_ini = np.tile(self.all_deltas[-1].reshape(-1,inputs.shape[0]) @ inputs,self.old_updates[0].shape)
        
        self.all_updates.append(self.train_step*(self.momentum*self.old_updates[0]-(1-self.momentum)*(new_info_ini)))
        #self.all_updates.append(self.train_step*(self.momentum*self.old_updates[0]-(1-self.momentum)*(self.all_deltas[-1].reshape(-1,inputs.shape[0]) @ inputs)))
        
        self.old_updates = self.all_updates.copy()
        # we want to write it backwards (see Backpropagation function)
        self.old_updates.reverse()
        
        
        for i in range(self.no_layers):
            # weights is written from left to right
            # updates are written from right to left
            self.all_weights[i] += self.all_updates[-(i+1)]
   
# In[3]:

def main(ds,neuron_topol,step,mom,eps):
    
    ### TRAIN
    Net = Perceptron(ds.X.shape[1],neuron_topol,step,mom,eps)
    Net.Train_NN(ds)
 
    ### PLOTTING
    y_til = np.zeros((ds.X.shape[0]))
    
    for pos,in1 in enumerate(ds.X):
        y_til[pos] = Net.Output(in1)
        
    plt.figure()
    plt.scatter(ds.X, ds.Y,color='blue')
    plt.plot(ds.X, y_til,color='red')
    plt.show()

 # In[4]:

if __name__ == "__main__":
    import time
    start_time = time.time()
    main(dataset, (3,2,1), 0.001, 0.9, 1e-06)
    print("--- %s seconds ---" % (time.time() - start_time))
    
