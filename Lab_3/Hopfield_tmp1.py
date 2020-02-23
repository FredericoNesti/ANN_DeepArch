import numpy as np
import itertools as it
import matplotlib.pyplot as plt

def sign(x):
    if x >= 0:
        return 1
    else:
        return -1


class Hopfield():
    def __init__(self, input_mem_patterns):
        self.n_nodes = input_mem_patterns.shape[1]
        self.weights = self.init_weights(input_mem_patterns)

    def init_weights(self, input_mem_patterns):
        """
            wij = 1/N sum from mu=1 to P of x_i^mu x_j^mu
        """
        weights = np.matmul(input_mem_patterns.T, input_mem_patterns)
        return weights / self.n_nodes  # good initialization
        # return np.random.normal(0, 1, (self.n_nodes,self.n_nodes))  # random init
        #return 0.5 * (np.random.normal(0, 1, (self.n_nodes, self.n_nodes)) +
        #              np.random.normal(0, 1, (self.n_nodes, self.n_nodes)))  # random, but symetric

    def update_rule(self, pattern):
        """
            Pertorms the update on weights as following:
                xi sign ( sum over j of wij xj )
        """
        return np.apply_along_axis(lambda t: sign(np.dot(t, pattern)), 1, self.weights)

    def update_till_convergence(self, pattern, follow_energy=False,return_iter=False):

        if follow_energy:
            energy = [self.energy(pattern)]

        next_pattern = self.update_rule(pattern)
        max_int_init = 200
        max_int = max_int_init
        while (not np.all(next_pattern == pattern) and max_int > 0):
            if follow_energy:
                energy.append(self.energy(next_pattern))

            pattern = next_pattern
            next_pattern = self.update_rule(pattern)
            max_int -= 1

        if follow_energy:
            interaction = [i for i in range(len(energy))]
            plt.plot(interaction, energy, label="Energy")
            plt.legend(bbox_to_anchor=(0.05, .95), loc='upper left', borderaxespad=0.)
            plt.show()
        #print("Iterations to convergence: " + str(200-max_int))
        if return_iter:
            return next_pattern, (max_int_init-max_int)
        return next_pattern

    def energy(self, pattern):
        """
            return the following energy function E = −sum_i sum_j wi*jx_i*x_j
        """
        return -np.dot(np.matmul(self.weights, pattern), pattern)


# input_mem_patterns = np.array([[-1, -1, 1, -1, 1, -1, -1, 1],
#                                [-1, -1, -1, -1, -1, 1, -1, -1],
#                                [-1, 1, 1, -1, -1, 1, -1, 1]])
#
# input_mem_patterns = np.array([[-1, -1, 1, -1, 1, -1, -1, 1],
#                                [1, 1, -1, 1, -1, 1, 1, -1]])

# Dissimilar paterns
# input_mem_patterns = np.array([[-1, -1,  1, -1,  1, -1, -1,  1],
#                               [ 1,  1, -1,  1, -1,  1,  1, -1]])

hp = Hopfield(input_mem_patterns)

## Task 3.1
# check for noise
# xd = np.array([[1, -1, 1, -1, 1, -1, -1, 1],
#                [1, 1, -1, -1, -1, 1, -1, -1],
#                [1, 1, 1, -1, 1, 1, -1, 1]])
#
# for x in xd:
#     print("Input pattern:", x)
#     print("Output pattern:", hp.update_till_convergence(x))
#     print("")
# attractors = set()
# diff = np.zeros(6)
# counter = np.zeros(6)
# for sample in list(it.product([-1, 1], repeat=8)):
#     min_d = np.inf
#     for stored in xd:
#         tmp = np.sum((stored == sample)*1)
#         if min_d > tmp:
#             min_d = tmp
#     min_d=min_d//1
#     ts, iterations_needed = hp.update_till_convergence(sample, return_iter=True)
#     diff[min_d] += iterations_needed
#     counter[min_d]+=1
#     attractors.add(np.array2string(ts))
# diff = diff/counter
# plt.plot(diff)
# plt.show()
# print("ATTRACTORS: n=" + str(len(attractors)))
# for at in attractors:
#     print(at)
# print("------")

## Task 3.2 Sequential Update
#patterns = np.loadtxt('pict.dat', dtype=int, delimiter=',').reshape(11,1024)
#print("")

## Task 3.3 Energy
# print("Task 3.3")

# print("Energy of distorted patterns")
# for pat in xd:
#     print("pattern:", pat, "Energy:", hp.energy(pat))

#
# print("Following energy with random weights")  # TODO: We must update the input sequentially as in Task 3.2
# for pat in xd:
#     hp.update_till_convergence(pat, follow_energy=True)
#
# print("Conclusion: the energy doesnt always decrease and doesnt converge")
#
#
# print("Following energy with random, but symetric, weights")
# for pat in xd:
#     hp.update_till_convergence(pat, follow_energy=True)
#
# print("Conclusion: it is unstable and can increase, but it converges")
