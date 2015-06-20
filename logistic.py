import numpy as np

class logistic:
    learning_rate = 0.01
    #constructor
    def __init__(self, n_units, n_inputs):
        print 'Initializing logistic layer with ' + str(n_units) + ' units and ' + str(n_inputs) + ' inputs per unit '
        self.n_outputs = n_units
        self.w = np.random.rand(n_inputs, self.n_outputs)/n_inputs - 0.5
        self.w0 = np.random.rand(self.n_outputs)/n_inputs - 0.5
    
    #take data from previous layer as input
    def forward(self, x):
        #store x for back-propagation
        self.x = x
        #estimate logit
        z = np.dot(self.w.T, x) + self.w0
        #estimate activation and store for back-propagation
        self.y = 1/(1 + np.exp(-z))
        return self.y
    
    #take error-derivative with respect to output as parameter
    def backprop(self, dEdy):
        self.dEdy = dEdy
        #estimate dE/dz = dE/dy * dy/dz, where
        #dy/dz = y(1-y)
        dydz = self.y*(1-self.y)
        dEdz = self.dEdy*dydz
        
        #estimate dE/dw = dE/dy * dy/dz * dz/dw, where
        # - dz/dw = x and
        # - dz/dw0 = 1
        # and sum over all training samples
        dEdw = np.dot(self.x, dEdz.T)
        dEdw0 = sum(dEdz.T)
        
        #update weights
        self.w += self.learning_rate * dEdw
        self.w0 += self.learning_rate * dEdw0
