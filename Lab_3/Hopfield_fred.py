# In[0]:
import numpy as np
import itertools as it
import matplotlib.pyplot as plt

def sign(x):
    if x >= 0:
        return 1
    else:
        return -1

def sign2(x):
    if x >= 0:
        return 1
    else:
        return 0
    
# In[0]:
        
### Hopfield Network
class Hopfield():
    def __init__(self, input_mem_patterns, bias, avg_activity=0, weights_init=''):
        self.avg_activity = avg_activity
        self.n_nodes = input_mem_patterns.shape[1]
        self.weights = self.init_weights(input_mem_patterns, weights_init)
        self.bias = bias
        
    def init_weights(self, input_mem_patterns, weights_init):
        """
            wij = 1/N sum from mu=1 to P of x_i^mu x_j^mu
            or random inicialization according to weights_init
        """
        if weights_init == 'random':  # random init
            return np.random.normal(0, 1, (self.n_nodes, self.n_nodes))

        elif weights_init == 'symetric_random':  # random, but symetric
            rand_weight = np.random.normal(0, 1, (self.n_nodes, self.n_nodes))
            return 0.5 * (rand_weight + rand_weight.T)

        else:
            weights = np.matmul(input_mem_patterns.T-self.avg_activity, input_mem_patterns-self.avg_activity)
        return weights / self.n_nodes  # good initialization


    def update_rule(self, pattern, random_seq_update=False, theta=0.5):
        """
            Pertorms the update on weights as following:
                xi sign ( sum over j of wij xj )
            if sequential updatate is true, we update only a few values of the pattern
        """
        new_pattern = np.apply_along_axis(lambda t: sign(np.dot(t, pattern)), 1, self.weights)
        if random_seq_update:
            rand_selection = np.random.rand(len(new_pattern)) < theta
            return 1*rand_selection * pattern + 1*(~rand_selection) * new_pattern
        else:
            return new_pattern

    def update_till_convergence(self, pattern, follow_energy=False, type_update='', verbose=False, max_int=1000):
        """
            Performs the update of the input pattern until convergence of until the max number of interactions.
        """
        energy = [self.energy(pattern)]
        my_pattern = np.copy(pattern)

        curr_int = 0
        converged = False
        while not converged and curr_int < max_int:
            prev_pattern = np.copy(pattern)  # it will be compared to pattern to check for convergence

            if type_update == 'seq_update':
                for i in range(len(pattern)):
                    pattern[i] = sign(np.dot(self.weights[i], pattern))
                    curr_int += 1
                    energy.append(self.energy(pattern))
                    if curr_int % 100 == 0 and verbose:
                        self.plot_binary_image(pattern, "seq_update interaction: " + str(curr_int))

            elif type_update == 'random_update':
                pattern = self.update_rule(pattern, random_seq_update=True)

            elif type_update == 'slightly_update': # question 3.6
                for i in range(len(pattern)):
                    my_pattern[i] = 0.5+0.5*sign2(np.dot(self.weights[i], my_pattern)-self.bias)
                    curr_int += 1
                    energy.append(self.energy(my_pattern))
                    if curr_int % 100 == 0 and verbose:
                        self.plot_binary_image(pattern, "seq_update interaction: " + str(curr_int))           
            
            else:
                pattern = self.update_rule(pattern)

            curr_int += 1
            energy.append(self.energy(pattern))
            converged = np.all(prev_pattern == pattern)

        if verbose:
            print("Total number of intetarions:", curr_int)

        if follow_energy:  # plot the graph of the energy
            interaction = [i for i in range(len(energy))]
            plt.plot(interaction, energy)
            plt.xlabel("Number of interactions")
            plt.ylabel("Energy")
            plt.show()

        return my_pattern

    def plot_binary_image(self, pattern, title=""):
        """
        Plot the binary image in pattern
        """
        plt.imshow(pattern.reshape(32, 32), cmap=plt.cm.gray)
        plt.title(title)
        plt.show()


    def energy(self, pattern):
        """
            return the following energy function E = −sum_i sum_j wi*jx_i*x_j
        """
        return -np.dot(np.matmul(self.weights, pattern), pattern)

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

def shuffle(vec,flipping_times):
    import numpy as np
    out_s = np.copy(vec)
    indexes_to_flip = [i for i in np.arange(vec.shape[0]).tolist()]                       
    ri = np.random.choice(indexes_to_flip,size=flipping_times,replace=False)
    for j,i in enumerate(ri):
        out_s[i] = -vec[i]
        #indexes_to_flip.remove(i)
    return out_s

# Hopfield for p1,p2 and p3 with noise

p1 = data_34[0,:]
p2 = data_34[1,:]
p3 = data_34[2,:]
p123 = np.vstack((p1,p2,p3))

hp1 = Hopfield(p123)


#p1_noisy = shuffle(data_34[0,:],100)
#hp1.plot_binary_image(p1_noisy)

# In[0]:

import time

iters = 1024
MC_iter = 200

gap = 40

res1 = []
res2 = []
res3 = []

ses1 = []
ses2 = []
ses3 = []

mes1 = []
mes2 = []
mes3 = []

start = time.time()
for i in range(1,iters+1,gap):
    print('---------------------------------------')
    print('Feature',i)
    r1 = 0
    r2 = 0
    r3 = 0
    
    s1 = 0
    s2 = 0
    s3 = 0
    
    m1 = 0
    m2 = 0
    m3 = 0

    for j in range(MC_iter):
        
        p1_noisy = shuffle(data_34[0,:],i)
        p2_noisy = shuffle(data_34[1,:],i)
        p3_noisy = shuffle(data_34[2,:],i)
        
        noise_conv1 = hp1.update_till_convergence(p1_noisy)
        noise_conv2 = hp1.update_till_convergence(p2_noisy)
        noise_conv3 = hp1.update_till_convergence(p3_noisy)
  
        r1 += np.all(p1==noise_conv1)
        r2 += np.all(p2==noise_conv2)
        r3 += np.all(p3==noise_conv3)
        
        s1 += np.sum(p1-noise_conv1)
        s2 += np.sum(p2-noise_conv2)
        s3 += np.sum(p3-noise_conv3)
        
        m1 += np.sum(~(p1==noise_conv1))
        m2 += np.sum(~(p2==noise_conv2))
        m3 += np.sum(~(p3==noise_conv3))
        
    res1.append(r1/MC_iter)
    res2.append(r2/MC_iter)
    res3.append(r3/MC_iter)
    
    ses1.append(s1/MC_iter)
    ses2.append(s2/MC_iter)
    ses3.append(s3/MC_iter)
    
    mes1.append(m1/MC_iter)
    mes2.append(m2/MC_iter)
    mes3.append(m3/MC_iter)
    

end = time.time()

elapsed = end - start
print('Time that took: ', elapsed)

# In[1]:

noise_perc = np.arange(1,iters+1,gap)/1024

plt.figure()
plt.plot(noise_perc,res1)
plt.plot(noise_perc,res2)
plt.plot(noise_perc,res3)
plt.show()

#plt.figure()
#plt.plot(noise_perc,ses1)
#plt.plot(noise_perc,ses2)
#plt.plot(noise_perc,ses3)
#plt.show()

plt.figure()
plt.plot(noise_perc,mes1)
plt.plot(noise_perc,mes2)
plt.plot(noise_perc,mes3)
plt.show()


# In[2]:

#p1_noisy = shuffle(data_34[0,:],100)
hp1.plot_binary_image(p1)
hp1.plot_binary_image(p2)
hp1.plot_binary_image(p3)

# In[3]:

#### 3.6:

import numpy as np

no_pat = 50
no_feat = 100

##########################
#change parameters here!
activity = 0.1
biases = np.arange(0,1,0.3)
MC_iter = 1
##########################


def create_random_patterns(activity,no_pat=no_pat,no_feat=no_feat):
    pattern_array = np.zeros((no_pat,no_feat))
    i=0
    while i < no_pat:
    #for i in range(no_pat):
        pattern_array[i,:] = np.random.binomial(1,activity,size=no_feat)
        if np.sum(np.abs(pattern_array[i,:] - pattern_array[0:i,:])) > i-1:
            i += 1
            
    return pattern_array 

patterns = create_random_patterns(activity)

#hp36 = Hopfield(patterns,avg_activity=activity,bias=b,type_update='slightly_update')


##### Test Bias

##########################
#change parameters here!
activity = 0.1
avg_activity = np.sum(patterns)/(no_feat*no_pat)
biases = np.array([0,0.5,1,2,10]) #np.arange(0,1,0.3)
MC_iter = 1000
##########################

out_store = np.zeros((no_pat,no_feat))
howmany_bias = np.zeros((biases.shape[0],no_pat))

for j,b in enumerate(biases):
    hp36 = Hopfield(patterns,avg_activity=avg_activity,bias=b)

    for _ in range(MC_iter):
        print(_)
        for i in range(no_pat):
            
            out_store[i,:] = hp36.update_till_convergence(patterns[i,:],type_update='slightly_update')
            howmany_bias[j,i] += np.all((out_store[i,:]==patterns[i,:]))
            
            #print(out_store)
            #print(howmany_bias)

plt.figure()
for j,b in enumerate(biases):
    plt.plot(np.arange(no_pat),howmany_bias[j,:]/MC_iter)
plt.xlabel('Bias')
plt.show()

##### Test Activity



