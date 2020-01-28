# In[0]:
#### Libraries
import numpy as np
import matplotlib.pyplot as plt

# In[1]:
#### Generate databases

x =  np.arange(0,10,0.5).reshape(-1,1)
y = np.random.normal(0,1,x.shape[0])

# In[2]:
#### ANNs

def Activation_Func(X_vec):
    #Sigmoid function
    return 1.0/(1.0 + np.exp(-X_vec)) 

def d_Activation_Func(X_vec):
    # Derivative of the Activation Function
    return (1+Activation_Func(X_vec))*(1-Activation_Func(X_vec))/2

def Activation_Output(X_vec):
    #Linear activation
    return X_vec

def d_Activation_Output(X_vec):
    # Derivative of the Activation Function for the last layer
    return np.ones((X_vec.shape))

class Perceptron():
    #without Bias
    #without momentum
    #without convergence check
       
    def __init__(self,input_dimensions,neurons_structure,train_step):
        
        self.train_step = train_step
        self.neurons_structure = neurons_structure
        self.no_inputs = input_dimensions
        self.no_layers = len(neurons_structure)
        self.Init_All_Weights()
        
    def Init_All_Weights(self):
        
        self.all_weights = []
        i_prevlayer = 0
        for pos, i in enumerate(self.neurons_structure):
            if pos == 0:
                # init weights (Normal 0 1)
                aux = np.random.normal(0,1,self.no_inputs*i).reshape(self.no_inputs,i)
            else:
                # init weights (Normal 0 1)
                aux = np.random.normal(0,1,i_prevlayer*i).reshape(i_prevlayer,i)
            i_prevlayer = i
            self.all_weights.append(aux)
        
    def Train_NN(self,inputs,outputs):
        
        self.Forward_step(inputs)
        self.Backprop_train(inputs,outputs)
        
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
        self.Forward_step(inputs)
        return Activation_Output(self.all_signals[-1])
        
    def Backprop_train(self,inputs,targets):
        # everything is in an opposite way
        o_ = Activation_Output(self.Output(inputs))
        d_o_ = d_Activation_Output(self.Output(inputs))
        
        self.all_deltas = [((o_-targets)*d_o_)]
        self.all_updates = []
        
        for i in range(self.no_layers-1,0,-1):
            
            self.all_updates.append(-self.train_step*( self.all_deltas[-1] @ self.all_signals[i] ))
            self.all_deltas.append(( self.all_deltas[-1] @ self.all_weights[i].T )*( d_Activation_Func(self.all_signals[i-1])))
            
        self.all_updates.append(-self.train_step*( self.all_deltas[-1].reshape(-1,inputs.shape[0]) @ inputs ))
        
        for i in range(self.no_layers):
            # weights is written from left to right
            # updates are written from right to left
            self.all_weights[i] += self.all_updates[-(i+1)]
   
# In[3]:
def main(IN,OUT,neuron_topol):
    
    ### TRAIN
    Net = Perceptron(IN.shape[1],neuron_topol,0.001)
    for pos,(in1,out1) in enumerate(zip(IN,OUT)):
        Net.Train_NN(in1,out1)
 
    ### PLOTTING
    y_til = np.zeros((IN.shape[0]))
    
    for pos,(in1,out1) in enumerate(zip(IN,OUT)):
        
        y_til[pos] = Net.Output(in1)
        
    plt.figure()
    plt.scatter(IN, OUT,color='blue')
    plt.plot(IN, y_til,color='red')
    plt.show()

 # In[4]:
if __name__ == "__main__":
    import time
    start_time = time.time()
    main(x, y, (3,2,1) )
    print("--- %s seconds ---" % (time.time() - start_time))
    
