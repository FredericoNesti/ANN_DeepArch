import numpy as np
import itertools as it
import matplotlib.pyplot as plt

def sign(x):
    if x >= 0:
        return 1
    else:
        return -1


class Hopfield():
    def __init__(self, input_mem_patterns, weights_init=''):
        self.n_nodes = input_mem_patterns.shape[1]
        self.weights = self.init_weights(input_mem_patterns, weights_init)

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
            weights = np.matmul(input_mem_patterns.T, input_mem_patterns)
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

    def update_till_convergence(self, pattern, follow_energy=False, type_update='', verbose=False, max_int=20000):
        """
            Performs the update of the input pattern until convergence of until the max number of interactions.
        """
        energy = [self.energy(pattern)]

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

        return pattern

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
