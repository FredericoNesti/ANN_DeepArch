import numpy as np
import itertools as it
import matplotlib.pyplot as plt


def sign(x, theta=0):
    #print(theta, "theta")
    if x >= theta:
        return 1
    else:
        return -1


def plot_binary_image(pattern):
        plt.imshow(pattern.reshape(32, 32), cmap=plt.cm.gray)
        plt.show()


class Hopfield:
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

    def update_rule(self, pattern, seq_update=False, theta=0):
        """
            Pertorms the update on weights as following:
                xi sign ( sum over j of wij xj )

            if sequential updatate is true, we update only a few values of the pattern
        """
        new_pattern = np.apply_along_axis(lambda t: 0.5 + 0.5 * sign(np.dot(t, pattern), theta=theta), 1, self.weights)
        if seq_update:
            n_updates = np.random.randint(len(new_pattern))
            r = np.random.random(len(new_pattern))
            return np.array([new_pattern[i] if rand < 0.5 else pattern for i,rand in enumerate(r)])
        else:
            return new_pattern

    def update_till_convergence(self, pattern, follow_energy=False, seq_update=False, theta=0):

        if follow_energy:
            energy = [self.energy(pattern)]

        next_pattern = self.update_rule(pattern, seq_update, theta=theta)
        max_int = len(pattern)//2
        while (not np.all(next_pattern == pattern) and max_int > 0):
            if follow_energy:
                energy.append(self.energy(next_pattern))

            pattern = next_pattern
            next_pattern = self.update_rule(pattern, seq_update, theta=theta)
            max_int -= 1

        if follow_energy:
            interaction = [i for i in range(len(energy))]
            plt.plot(interaction, energy, label="Energy")
            plt.legend(bbox_to_anchor=(0.05, .95), loc='upper left', borderaxespad=0.)
            plt.show()

        return next_pattern


    def energy(self, pattern):
        """
            return the following energy function E = −sum_i sum_j wi*jx_i*x_j
        """
        return -np.dot(np.matmul(self.weights, pattern), pattern)

np.random.seed(100)

incl = 100  # samples to test
dims = 100  # dimensions to use
ratio = 0.05  # ρ
thetas_to_test = np.arange(0, 10, 0.1)  # θ range and scale
repeat_nums =10  # number of random samples to test
threshold = 1
patterns = np.zeros((incl, dims))
for i in range(incl):
    patterns[i, np.random.choice(dims, int(ratio*dims), replace=False)] = 1
#patterns = np.eye(dims)
saving = []
for thetas in thetas_to_test:
    np.random.seed(10)
    for n in range(1, incl+1):
        not_attr = 0
        for rnd_iter in range(repeat_nums):
            pats = patterns[np.random.choice(patterns.shape[0], n, replace=False)]
            hp = Hopfield((pats-ratio))
            hp.weights = hp.weights*dims
            for pat in pats:
                test_p = hp.update_till_convergence(pat, theta=thetas)
                if not np.all(test_p == pat):
                    not_attr += 1
        print(n, not_attr)
        if not_attr > threshold:
            saving.append(n-1)
            break
        if n == incl:
            saving.append(n)
plt.plot(thetas_to_test, saving)
arg_max = np.argmax(saving)
_max = np.max(saving)
plt.plot([thetas_to_test[arg_max], thetas_to_test[arg_max]], [0, _max], 'k--')
tmp = np.arange(np.min(thetas_to_test), np.max(thetas_to_test) + 0.01, 1)
tmp = np.append(tmp, thetas_to_test[arg_max])
tmp.sort()
print(tmp)
plt.xticks(tmp)
plt.xlabel("θ")
plt.ylabel("Capacity")
plt.show()



