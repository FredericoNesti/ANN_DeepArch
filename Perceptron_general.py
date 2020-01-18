# In[0]:
#### Libraries
import numpy as np
import matplotlib.pyplot as plt



# In[1]:
#### Generate databases

x =  np.arange(0,10,0.5)
y = np.random.normal(0,1,x.shape[0])


# In[2]:
#### ANNs

def Activation_Func(X_vec):
    #Sigmoid function
    import numpy as np
    return 1.0/(1.0 + np.exp(-X_vec)) 

def Activation_Output(X_vec):
    #Linear activation
    return X_vec

class Perceptron():
    #without Bias
    
    def __init__(self,inputs,targets,neurons_structure):
        
        self.neurons_struct = neurons_structure
        self.no_layers = len(neurons_structure)
        #self.inputs = inputs
        self.targets = targets
        self.no_inputs = 1#inputs[0]
        self.inputs = inputs.reshape(-1,self.no_inputs)
        
        self.all_weights = []
        i_prevlayer = 0
        for pos, i in enumerate(neurons_structure):
            if pos == 0:
                # init weights (Normal 0 1)
                aux = np.random.normal(0,1,self.no_inputs*i).reshape(self.no_inputs,i)
            else:
                # init weights (Normal 0 1)
                aux = np.random.normal(0,1,i_prevlayer*i).reshape(i_prevlayer,i)
            i_prevlayer = i
            self.all_weights.append(aux)
        
    
    def Forward_step(self):
        #Propagating massage (linear combinations)
        self.all_signals = []
        i_prevlayer = 0
        for i in range(self.no_layers):
            if i == 0:
                self.all_signals.append( self.all_weights[i].T @ self.inputs )
            else:
                self.all_signals.append( self.all_weights[i].T @ Activation_Func(self.all_signals[i_prevlayer]) )
            i_prevlayer = i
            
    def Output(self):
        self.Forward_step()
        return Activation_Output(self.all_signals[-1])
    
    #def Functional_Error():
        
    #def Backprop_train():
   
# In[3]:
def main(IN,OUT,neuron_topol):
    
    #ANN = Perceptron(IN, OUT, (3,2,1) )
    
    y_til = np.zeros((IN.shape[0]))
    
    #print(IN.shape[0])
    #print('checkpoint')
    #print(y_til.shape)
    
    for pos,(in1,out1) in enumerate(zip(IN,OUT)):
        #print(pos)
        #print(in1)
        #print(out1)
        
        Net = Perceptron(in1, out1, neuron_topol)
        
        #print(Net.Output())
        
        y_til[pos] = Net.Output()[0][0]  
        
    plt.figure()
    plt.scatter(IN, OUT,color='blue')
    plt.plot(IN, y_til,color='red')
    #np.array(list(map(Perceptron, IN, OUT)))
    plt.show()

# In[4]:
if __name__ == "__main__":
    import time
    start_time = time.time()
    main(x, y, (3,2,1) )
    print("--- %s seconds ---" % (time.time() - start_time))
    

