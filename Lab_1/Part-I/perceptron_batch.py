import numpy as np
from dataset_class_norm import Dataset
class Perceptron():
    def __init__(self,dims,rate = 1):
        self.weight = np.random.rand(1,dims)
        self.rate = rate
    def train(self,inputs,outputs):
        change = np.zeros(self.weight.shape)
        for i,sample in enumerate(inputs):
            sample_re = sample.reshape(-1,1)
            calc = np.sum(self.weight @ sample_re)
            classified = (calc>0)*2-1
            change += (classified - outputs[i])/2*sample_re.T
        self.weight += -self.rate*change/inputs.shape[0]
        return np.sum(change)
    def batch_method(self,database, max_iter = 200):
        num_of_splits = int(2*database.n/database.batchSize)
        counter = 0
        iterations=0
        while(counter<num_of_splits and iterations<max_iter):
            inputs, outputs = database.nextBatch()
            inputs = np.insert(inputs,-1,1,axis=1)
            error = self.train(inputs,outputs)
            #print(error)
            if error == 0:
                counter+=1
            else:
                counter = 0
            iterations+=1
        print("Iterations:" + str(iterations))
