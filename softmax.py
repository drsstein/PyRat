# short lecture on learning linear neurons by Geoffrey Hinton:
# https://www.youtube.com/watch?v=yqsI-X40OBY

import numpy as np

class softmax:

    def __init__(self, n_units, n_inputs):
        self.n_outputs = n_units
        self.n_inputs = n_inputs
        self.w = (np.random.rand(self.n_inputs+1, self.n_outputs)- 0.5)/self.n_inputs 
    
    def forward(self, x):
        self.x = np.vstack([x, np.ones(x.shape[1])])

        #estimate logit
        z = np.dot(self.w.T, self.x)

        #estimate output
        # y_i = e^(z_i) / sum_i(e^(z_i))
        self.y = np.exp(z)
        self.y /= sum(self.y)
        return self.y
    
    #as the softmax is always the last layer in the network, and the cross-entropy
    #error is computed with respect to the logit (y - t), we do both
    #forward and backpropagation in a single function call
    def evaluate(self, x, targets, learning_rate):
        self.forward(x)
        
        #estimate error derivative
        dEdz = targets - self.y
        #estimate cross entropy error across samples
        #it is important to normalize here wrt. the number of training cases
        n_samples = dEdz.shape[1]
        C = sum(-sum(targets*np.log(self.y)))/n_samples
        
        #propagate backwards through weights and inputs
        dEdw = np.dot(self.x, dEdz.T)/n_samples
        
        #dE/dx for each training sample
        dEdx = np.dot(self.w[0:self.n_inputs,:], dEdz)
        
        #update weights
        self.w += learning_rate * dEdw
        
        return C, dEdx
        
