import numpy as np

def sign(x):
    if x >= 0: return 1
    else: return -1

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
        while(np.all(next_pattern != pattern)):
            pattern = next_pattern
            next_pattern = self.update_rule(pattern)

        return next_pattern

input_mem_patterns = np.array([[-1, -1,  1, -1,  1, -1, -1,  1],
                               [-1, -1, -1, -1, -1,  1, -1, -1],
                               [-1,  1,  1, -1, -1,  1, -1,  1]])

hp = Hopfield(input_mem_patterns)


## Task 3.1
# check for noise
xd = np.array([[1, -1,  1, -1,  1, -1, -1,  1],
               [1,  1, -1, -1, -1,  1, -1, -1],
               [1,  1,  1, -1,  1,  1, -1,  1]])

for x in xd:
    print("Input pattern:", x)
    print("Output pattern:", hp.update_till_convergence(x))
    print("")

print(np.linalg.eig(hp.weights))