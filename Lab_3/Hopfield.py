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
        # return 0.5 * (np.random.normal(0, 1, (self.n_nodes, self.n_nodes)) +
        #               np.random.normal(0, 1, (self.n_nodes, self.n_nodes)))  # random, but symetric

    def update_rule(self, pattern, seq_update=False):
        """
            Pertorms the update on weights as following:
                xi sign ( sum over j of wij xj )

            if sequential updatate is true, we update only a few values of the pattern
        """
        new_pattern = np.apply_along_axis(lambda t: sign(np.dot(t, pattern)), 1, self.weights)
        if seq_update:
            n_updates = np.random.randint(len(new_pattern))
            r = np.random.random(len(new_pattern))
            return np.array([new_pattern[i] if rand < 0.5 else pattern for i,rand in enumerate(r)])
        else:
            return new_pattern

    def update_till_convergence(self, pattern, follow_energy=False, seq_update=False):

        if follow_energy:
            energy = [self.energy(pattern)]

        next_pattern = self.update_rule(pattern, seq_update)
        max_int = 20
        while (not np.all(next_pattern == pattern) and max_int > 0):
            if follow_energy:
                energy.append(self.energy(next_pattern))

            pattern = next_pattern
            next_pattern = self.update_rule(pattern, seq_update)
            max_int -= 1

        if follow_energy:
            interaction = [i for i in range(len(energy))]
            plt.plot(interaction, energy, label="Energy")
            plt.legend(bbox_to_anchor=(0.05, .95), loc='upper left', borderaxespad=0.)
            plt.show()

        return next_pattern

    def plot_binary_image(self, pattern):
        plt.imshow(pattern.reshape(32, 32), cmap=plt.cm.gray)
        plt.show()


    def energy(self, pattern):
        """
            return the following energy function E = −sum_i sum_j wi*jx_i*x_j
        """
        return -np.dot(np.matmul(self.weights, pattern), pattern)


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

# for x in xd:
#     print("Input pattern:", x)
#     print("Output pattern:", hp.update_till_convergence(x))
#     print("")
# attractors = set()
# for sample in list(it.product([-1, 1], repeat=8)):
#     ts = hp.update_till_convergence(sample)
#     attractors.add(np.array2string(ts))
# print("ATTRACTORS: n=" + str(len(attractors)))
# for at in attractors:
#     print(at)
# print("------")

## Task 3.2 Sequential Update
patterns = np.loadtxt('pict.dat', dtype=int, delimiter=',').reshape(11,1024)
hp = Hopfield(patterns[:3])  # three first elements are learned

for i,el in enumerate(patterns[:3]):  # check that the tree paterns are stable
    print("patern", i, np.all(el == hp.update_rule(el)))

print("Conclusion: yes, they are atractors")


print("Degraded pattern")
# hp.plot_binary_image(patterns[0])  # p1
# hp.plot_binary_image(patterns[9])  # p10, a degraded version of p1
# hp.plot_binary_image(hp.update_till_convergence(patterns[9]))
#
# hp.plot_binary_image(patterns[1])  # p2
# hp.plot_binary_image(patterns[2])  # p3
# hp.plot_binary_image(patterns[10])  # p11, a degraded version of p2 and p3
# hp.plot_binary_image(hp.update_till_convergence(patterns[10]))

print("Conclusion: no, p10 converges to p1, but p11 got lost")

print("Degraded pattern with sequential update")
hp.plot_binary_image(hp.update_till_convergence(patterns[10], seq_update=True))

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
