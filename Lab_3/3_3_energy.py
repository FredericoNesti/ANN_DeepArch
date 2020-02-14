# Task 3.3 Energy

import numpy as np
import itertools as it
from Hopfield import Hopfield

input_mem_patterns = np.array([[-1, -1, 1, -1, 1, -1, -1, 1],
                               [-1, -1, -1, -1, -1, 1, -1, -1],
                               [-1, 1, 1, -1, -1, 1, -1, 1]])

hp = Hopfield(input_mem_patterns)

## Task 3.1
# check for noise
xd = np.array([[1, -1, 1, -1, 1, -1, -1, 1],
               [1, 1, -1, -1, -1, 1, -1, -1],
               [1, 1, 1, -1, 1, 1, -1, 1]])

print("Energy of the different attractors")
attractors = set()
for sample in list(it.product([-1, 1], repeat=8)):
    ts = hp.update_till_convergence(sample)
    attractors.add(np.array2string(np.array(ts)))

for att in attractors:
    att = att.replace('[', '')
    att = att.replace(']', '')
    att = np.fromstring(att, dtype=int, sep=" ")
    print(np.array2string(att), "energy: ", hp.energy(att))

print("\nEnergy of distorted patterns")
for pat in xd:
    print("pattern:", pat, "Energy:", hp.energy(pat))

print("\nFollowing energy for sequential update")
# OBS: I did it with the images for having more time to convergence
patterns = np.loadtxt('pict.dat', dtype=int, delimiter=',').reshape(11, 1024)
hp = Hopfield(patterns[:3])  # three first elements are learned
hp.update_till_convergence(patterns[10], type_update="seq_update", follow_energy=True)


print("\nFollowing energy with random weights")
hp = Hopfield(patterns[:3], weights_init='random')  # three first elements are learned
hp.update_till_convergence(patterns[10], type_update="seq_update", follow_energy=True)

print("Conclusion: the energy doesnt always decrease and doesnt converge")

print("\nFollowing energy with random weights, but symetric")
hp = Hopfield(patterns[:3], weights_init='symetric_random')  # three first elements are learned
hp.update_till_convergence(patterns[10], type_update="seq_update", follow_energy=True)


print("Conclusion: even with random inicialization, it converges smoothly and monotonically")
