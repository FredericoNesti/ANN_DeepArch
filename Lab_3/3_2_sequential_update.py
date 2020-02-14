## Task 3.2 Sequential Update

import numpy as np
import itertools as it
from Hopfield import Hopfield

patterns = np.loadtxt('pict.dat', dtype=int, delimiter=',').reshape(11, 1024)
hp = Hopfield(patterns[:3])  # three first elements are learned

for i,el in enumerate(patterns[:3]):  # check that the tree paterns are stable
    print("patern", i, np.all(el == hp.update_rule(el)))

print("Conclusion: yes, they are atractors\n")


print("Degraded pattern")
hp.plot_binary_image(patterns[0], "P1")  # p1
hp.plot_binary_image(patterns[9], "P10")  # p10, a degraded version of p1
hp.plot_binary_image(hp.update_till_convergence(patterns[9]), "Result synchronous P10")

hp.plot_binary_image(patterns[1], "P2")  # p2
hp.plot_binary_image(patterns[2], "P3")  # p3
hp.plot_binary_image(patterns[10], "P11")  # p11, a degraded version of p2 and p3
hp.plot_binary_image(hp.update_till_convergence(patterns[10]), "Result synchronous P11")

print("Conclusion: the input pattern not always converge to p1, p2 and p3."
      "p10 converges to p1, but p11 got lost")

print("Degraded pattern with random update")
hp.plot_binary_image(hp.update_till_convergence(patterns[10], type_update="random_update", verbose=True), "Result random P11")

print("Degraded pattern with sequential update")
hp.plot_binary_image(hp.update_till_convergence(patterns[10], type_update="seq_update", verbose=True), "Result sequential P11")