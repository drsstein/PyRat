import matplotlib.pyplot as plt
import numpy as np

from numpy.random import rand

def test():
    #generate n samples of random data
    n = 100
    x = rand(n)

    #transform data linearly
    a = 2
    b = 0
    y = a*x + b
    #add some white noise
    y += rand(len(x)) - 0.5
    
    #reshape x to contain each sample as one vector
    x = x.reshape(1, len(x))

    #make y 't'arget for prediction
    t = y
    #define number of iterations
    n_iter = 100
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
        error = sum(dy)
        
        #multiply with dy/dw
        dw = np.dot(x, dy)
        
        #update weights
        w += nu * dw
        w0 += nu * error
        print "mean dE:" + str(error) + " w0: " + str(w0) + " w: " + str(w)

    l_x = np.linspace(0, 1, 100).reshape(x.shape)
    l_y = np.dot(w.T, l_x) + w0
    plt.scatter(x, t)
    plt.plot(l_x.T, l_y)
    plt.ioff()
    plt.show()
