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

    for i in range(n_iter):
        #feed data through linear neuron
        y = np.dot(w.T, x) + w0
        
        #estimate dE/dy, where E is squared error
        dy = t - y
        
        #multiply with dy/dw and sum over training samples
        dw = np.dot(x, dy)
        #dw for w0 is dy*1 = dy
        dw0 = sum(dy)
        
        #update weights
        w += nu * dw
        w0 += nu * dw0
        
        #print error and weights at end of each iteration
        print "mean dE/dy:" + str(dw0) + " w0: " + str(w0) + " w: " + str(w)
    
    #generate line from learned weights
    l_x = np.linspace(0, 1, 2).reshape(x.shape)
    l_y = np.dot(w.T, l_x) + w0
    #plot training samples and fitted line
    plt.scatter(x, t)
    plt.plot(l_x.T, l_y)
    plt.show()
