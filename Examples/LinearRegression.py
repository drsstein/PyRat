# short lecture on learning linear neurons by Geoffrey Hinton:
# https://www.youtube.com/watch?v=WqQivCl8dmQ

import matplotlib.pyplot as plt
import numpy as np

from numpy.random import rand

def run():
    #generate n samples of random data
    n = 100
    x = rand(n)

    #transform data linearly
    a = 2
    b = 0
    y = a*x + b
    #add some white noise
    y += rand(len(x)) - 0.5
    
    #reshape x to contain samples as column vectors (one row per input dimension)
    x = x.reshape(1, len(x))

    #make y 't'arget for prediction
    t = y
    
    #define number of training epochs (look at each example once per epoch)
    n_epochs = 100
    #define learning rate
    nu = 0.01

    #randomly initialise weights
    n_dims = len(x)
    w = rand(n_dims)*0.1 - 0.5
    w0 = rand(1)*0.1 - 0.5

    for i in range(n_epochs):
        #feed data through linear neuron
        y = np.dot(w.T, x) + w0
        
        #estimate dE/dy, where E is squared error
        dEdy = t - y
        
        #multiply dE/dy with dy/dw and sum over training samples
        # - dy/dw = x and
        # - dy/dw0 = 1
        dEdw = np.dot(x, dEdy)
        dEdw0 = sum(dEdy)
        
        #update weights
        w += nu * dEdw
        w0 += nu * dEdw0
        
        #print error and weights at end of each iteration
        print "sum dE/dy:" + str(dEdw0) + " w0: " + str(w0) + " w: " + str(w)
    
    #generate line from learned weights
    l_x = np.linspace(0, 1, 2).reshape(1, -1)
    l_y = np.dot(w.T, l_x) + w0
    #plot training samples and fitted line
    plt.scatter(x, t)
    plt.plot(l_x.T, l_y)
    plt.show()
