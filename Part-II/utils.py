import numpy as np
import math
import matplotlib.pyplot as plt

from collections import defaultdict


def generateDataset(start_t, end_t, beta=0.2, gamma=0.1, n=10, tal=25, noise = False, sigma_noise=0.03):
    x = defaultdict(float)  # represent a function
    x[0] = 1.5
    for t in range(1,end_t+20):
        x[t + 1] = x[t] + (beta * x[t - tal]) / (1 + math.pow(x[t - tal], n)) - gamma * x[t]

    if noise:
        for t in range(1, end_t + 20):
            x[t + 1] += np.random.normal(0, sigma_noise)

    input  = [[x[t-20], x[t-15], x[t-10], x[t-5], x[t]] for t in range(start_t, end_t)]
    output = [x[t+5] for t in range(start_t, end_t)]

    return input, output


def splitDataset(input, output):
    input_train , input_val , input_test  = input[:500] , input[500:1000] , input[1000:]
    output_train, output_val, output_test = output[:500], output[500:1000], output[1000:]

    return input_train , input_val , input_test, output_train, output_val, output_test


def plotPredictios(y_pred, y_test, n_input):
    t = [i for i in range(1000, n_input)]
    plt.plot(t, y_pred, label="Prediction")
    plt.plot(t, y_test, label="True values")
    plt.title('Predictions NN', fontweight="bold")
    plt.legend(bbox_to_anchor=(0.05, .95), loc='upper left', borderaxespad=0.)
    plt.show()


def plotComparePredictions(y_pred1, y_pred2, y_test, n_input, name_model1, name_model2):
    t = [i for i in range(1000, n_input)]
    plt.plot(t, y_pred1, label="Prediction " + name_model1)
    plt.plot(t, y_pred2, label="Prediction " + name_model2)
    plt.plot(t, y_test, label="True values")
    plt.title('Predictions NN', fontweight="bold")
    plt.legend(bbox_to_anchor=(0.05, .95), loc='upper left', borderaxespad=0.)
    plt.show()


def plotWeights(model, regularization_const):
    weights = model.get_weights()

    for w in weights:
        plt.hist(w.flatten(), bins=10, color='red')

    plt.title('Weights distribution NN - Regularizations const: ' + str(regularization_const), fontweight="bold")
    plt.xlabel('Weight value')
    plt.ylabel('Number of weights')
    plt.show()
