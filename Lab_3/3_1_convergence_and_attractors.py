## Task 3.1

import numpy as np
import itertools as it
from Hopfield import Hopfield

input_mem_patterns = np.array([[-1, -1, 1, -1, 1, -1, -1, 1],
                               [-1, -1, -1, -1, -1, 1, -1, -1],
                               [-1, 1, 1, -1, -1, 1, -1, 1]])

# Dissimilar paterns (study for more ortoghonal patterns)
# input_mem_patterns = np.array([[-1, -1,  1, -1,  1, -1, -1,  1],
#                               [ 1,  1, -1,  1, -1,  1,  1, -1]])

hp = Hopfield(input_mem_patterns)


# check for noise
xd = np.array([[1, -1, 1, -1, 1, -1, -1, 1],
               [1, 1, -1, -1, -1, 1, -1, -1],
               [1, 1, 1, -1, 1, 1, -1, 1]])

for x in xd:
    print("Input pattern:", x)
    print("Output pattern:", hp.update_till_convergence(x))
    print("")
attractors = set()
for sample in list(it.product([-1, 1], repeat=8)):
    ts = hp.update_till_convergence(sample)
    attractors.add(np.array2string(np.array(ts)))
print("ATTRACTORS: n=" + str(len(attractors)))
for at in attractors:
    print(at)
print("------")