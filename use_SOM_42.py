import os
os.chdir('C:/Users/frede/Desktop/Academic/KTH/ANN/Lab_2')

from SOM_circular import SOM
import numpy as np

#np.random.seed(143)

som = SOM('cities2.txt')
som.train()
som.input

output = np.zeros((10,10))
#output[:,0] = som.use(som.input[0]).reshape(1,-1)
for i in range(len(som.input)):
    output[:,i] = som.use(som.input[i]).reshape(1,-1)

del(som)

output
