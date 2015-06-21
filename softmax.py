# short lecture on learning linear neurons by Geoffrey Hinton:
# https://www.youtube.com/watch?v=yqsI-X40OBY

import numpy as np

class softmax:

    def __init__(self, n_units, n_inputs):
        self.n_outputs = n_units
        self.n_inputs = n_inputs
        self.w = np.random.rand(self.n_inputs+1, self.n_outputs)/self.n_inputs - 0.5
    
    #as the softmax is always the last layer in the network, and the cross-entropy
    #error can be easily computed with respect to the logit (y - t), we do both
    #forward and backpropagation in a single function call
    def evaluate(self, x, targets, learning_rate):
        self.x = np.vstack([x, np.ones(x.shape[1])])

        #estimate logit
        z = np.dot(self.w.T, self.x)

        #estimate output
        # y_i = e^(z_i) / sum_i(e^(z_i))
        self.y = np.exp(z)
        self.y /= sum(self.y)
        
        #estimate error derivative
        dEdz = self.y - targets
        #estimate cross entropy error across samples
        C = sum(-sum(targets*np.log(self.y)))
        
        #propagate backwards through weights and inputs
        dEdw = np.dot(self.x, dEdz.T)
        
        #dE/dx for each training sample
        dEdx = np.dot(self.w[0:self.n_inputs,:], dEdz)
        
        #update weights
        self.w += learning_rate * dEdw
        
        return C, dEdx
        
