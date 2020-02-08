# In[0]:
import numpy as np
import itertools as it

def sign(x):
    if x >= 0:
        return 1
    else:
        return -1
    
# In[0]:
        
### Hopfield Network

class Hopfield():
    def __init__(self, input_mem_patterns):
        self.n_nodes = input_mem_patterns.shape[1]
        self.weights = self.init_weights(input_mem_patterns)

    def init_weights(self, input_mem_patterns):
        """
            wij = 1/N sum from mu=1 to P of x_i^mu x_j^mu
        """
        weights = np.matmul(input_mem_patterns.T, input_mem_patterns)
        return weights / self.n_nodes

    def update_rule(self, pattern):
        """
        Pertorms the update on weights as following:
            xi sign ( sum over j of wij xj )
        """
        return np.apply_along_axis(lambda t: sign(np.dot(t, pattern)), 1, self.weights)

    def update_till_convergence(self, pattern):
        next_pattern = self.update_rule(pattern)
        while (np.all(next_pattern != pattern)):
            pattern = next_pattern
            next_pattern = self.update_rule(pattern)

        return next_pattern

# In[0]:

### Data for task 3.1

input_mem_patterns = np.array([[-1, -1, 1, -1, 1, -1, -1, 1],
                               [-1, -1, -1, -1, -1, 1, -1, -1],
                               [-1, 1, 1, -1, -1, 1, -1, 1]])

# Dissimilar paterns
# input_mem_patterns = np.array([[-1, -1,  1, -1,  1, -1, -1,  1],
#                               [ 1,  1, -1,  1, -1,  1,  1, -1]])

hp = Hopfield(input_mem_patterns)

## Task 3.1
# check for noise
xd = np.array([[1, -1, 1, -1, 1, -1, -1, 1],
               [1, 1, -1, -1, -1, 1, -1, -1],
               [1, 1, 1, -1, 1, 1, -1, 1]])

# In[0]:

# task 3.1    
    
for x in xd:
    print("Input pattern:", x)
    print("Output pattern:", hp.update_till_convergence(x))
    print("")
attractors = set()
for sample in list(it.product([-1, 1], repeat=8)):
    ts = hp.update_till_convergence(sample)
    attractors.add(np.array2string(ts))
print("ATTRACTORS: n=" + str(len(attractors)))
for at in attractors:
    print(at)
print("------")

# In[0]:

# Data 2 

import os 
os.chdir('C:/Users/frede/Desktop/Academic/KTH/ANN/Lab_3')

def load_input(filename, fdim):
    with open(filename) as f:
        #patterns = np.loadtxt((x.replace(';', ',') for x in f), dtype=str, delimiter=',', comments='%')
        patterns = np.loadtxt((x.replace(';', ',') for x in f), dtype=int, delimiter=',', comments='%')
        #patterns = patterns[:,[0,1]].astype(np.float)
        #print(patterns)
    return patterns.reshape(fdim)


data_34 = load_input('pict.dat',(11,1024))
data_34


# In[0]:
# Task 3.4: Distortion Resistance

def shuffle(vec,times):
    import numpy as np
    out_s = np.copy(vec)
    for i in range(times):
        ri = np.random.choice(np.arange(vec.shape[0]),size=2,replace=False)
        #print('shuffled indexes: ',ri)
        out_s[ri] = -vec[ri]

    return out_s

# Hopfield for p1,p2 and p3 with noise
def retrive_qtd_attractors(data,model,iters,MC_iter):
    
    p1 = data_34[0,:]
    p2 = data_34[1,:]
    p3 = data_34[2,:]
    p123 = np.vstack((p1,p2,p3))
    
    nn = model(p123)
    
    res1 = []
    res2 = []
    res3 = []
    
    for i in range(iters+1):        
        for j in range(MC_iter):
            
            p1_noisy = shuffle(data[0,:],iters)
            p2_noisy = shuffle(data[1,:],iters)
            p3_noisy = shuffle(data[2,:],iters)
            
            res1.append(np.sum(np.abs(p1_noisy-nn.update_till_convergence(p1_noisy)))==0)
            res2.append(np.sum(np.abs(p2_noisy-nn.update_till_convergence(p2_noisy)))==0)
            res3.append(np.sum(np.abs(p3_noisy-nn.update_till_convergence(p3_noisy)))==0)
    
    return res1,res2,res3

output = retrive_qtd_attractors(data_34,Hopfield,1024,100)

# In[1]:

o0 = np.mean(np.array(output[0]).reshape((1024,100)),axis=1)
o1 = np.mean(np.array(output[1]).reshape((1024,100)),axis=1)
o2 = np.mean(np.array(output[2]).reshape((1024,100)),axis=1)

noise_perc = np.arange((1025))/1024

import matplotlib.plot as plt
plt.plot(noise_perc,o0)
plt.plot(noise_perc,o1)
plt.plot(noise_perc,o2)



