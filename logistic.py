import numpy as np

class logistic:
    
    #constructor
    def __init__(self, n_units, n_inputs):
        print 'Initializing logistic layer with ' + str(n_units) + ' units and ' + str(n_inputs) + ' inputs per unit '
        self.n_outputs = n_units
        self.n_inputs = n_inputs
        self.w = np.random.rand(self.n_inputs+1, self.n_outputs)/self.n_inputs - 0.5
    
    #take data from previous layer as input
    def forward(self, x):
        #store x for back-propagation
        self.x = np.vstack([x, np.ones(len(x[0]))])
        #estimate logit
        z = np.dot(self.w.T, self.x)
        #estimate activation and store for back-propagation
        self.y = 1/(1 + np.exp(-z))
        return self.y
    
    #take error-derivative with respect to output as parameter
    def backprop(self, dEdy, learning_rate):
        self.dEdy = dEdy
        #estimate dE/dz = dE/dy * dy/dz, where
        #dy/dz = y(1-y)
        dydz = self.y*(1-self.y)
        dEdz = self.dEdy*dydz
        
        #estimate dE/dw = dE/dy * dy/dz * dz/dw, where
        # - dz/dw = x and
        # and sum over all training samples
        dEdw = np.dot(self.x, dEdz.T)
        
        #dE/dx for each training sample
        dEdx = np.dot(self.w[0:self.n_inputs,:], dEdz)
        
        #update weights
        self.w += learning_rate * dEdw
        
        return dEdx
        
        
