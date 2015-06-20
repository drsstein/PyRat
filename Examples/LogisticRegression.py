# short lecture on learning logistic neurons by Geoffrey Hinton:
# https://www.youtube.com/watch?v=--_F0rbPH9M

import math

import matplotlib.pyplot as plt
import numpy as np

def run():
    #generate n samples of random 2d data from each class
    n = 100
    x_pos = np.random.randn(2, n)
    y_pos = np.ones(n)
    x_neg = np.random.randn(2, n)
    y_neg = np.zeros(n)
    #shift negative data to center around m_neg
    m_neg = np.array([2, 1]).reshape(2, -1)
    x_neg += m_neg
    #concatenate positive and negative examples
    x = np.concatenate((x_pos, x_neg), axis=1)
    y = np.concatenate((y_pos, y_neg))
    #set y as 't'arget
    t = y
    
    #define number of training epochs (look at each example once per epoch)
    n_epochs = 200
    #define learning rate
    nu = 0.01

    #randomly initialise weights
    n_dims = len(x)
    n_samples = len(x[0])
    w = np.random.rand(n_dims)/n_dims - 0.5
    w0 = np.random.rand(1)/n_dims - 0.5
    rmse = np.zeros(n_epochs)
    
    for i in range(n_epochs):
        #feed data through logistic neuron
        z = np.dot(w.T, x) + w0
        y = 1/(1 + np.exp(-z))
        
        #estimate dE/dy, where E is squared error
        dEdy = t - y
        #log mean error
        rmse[i] = np.sqrt(np.dot(dEdy.T,dEdy)/n_samples)
        
        #estimate dE/dz = dE/dy * dy/dz, where
        #dy/dz = y(1-y)
        dydz = y*(1-y)
        dEdz = dEdy*dydz
        
        #estimate dE/dw = dE/dy * dy/dz * dz/dw, where
        # - dz/dw = x and
        # - dz/dw0 = 1
        # and sum over all training samples
        dEdw = np.dot(x, dEdz)
        dEdw0 = sum(dEdz)
        
        #update weights
        w += nu * dEdw
        w0 += nu * dEdw0
        
        #print error and weights at end of each iteration
        print "RMSE:" + str(rmse[i]) + " w0: " + str(w0) + " w: " + str(w)
    
    #plot data
    plt.figure(figsize=(20, 9))
    plt.subplot(121)
    plt.title('Separation in feature space')
    plt.scatter(x_pos[0,:], x_pos[1,:], color='blue', marker='+')
    plt.scatter(x_neg[0,:], x_neg[1,:], color='red', marker='o')
    
    #separation line at p=0.5 is defined as x2 = -w1/w2 * x1 - w0/w2
    b = -w0/w[1]
    a = -w[0]/w[1]
    l_x = np.linspace(-3, 4, 2)
    l_y = -w[0]/w[1] * l_x -w0/w[1]
    #separation line at probability p=0.25
    l_y1 = -w[0]/w[1] * l_x - (w0 + np.log(1/0.25-1))/w[1]
    #separation line at probability p=0.75
    l_y2 = -w[0]/w[1] * l_x - (w0 + np.log(1/0.75-1))/w[1]
    
    #plot separation lines
    plt.plot(l_x, l_y, color='black')
    plt.plot(l_x, l_y1, linestyle='dashed', color='red')
    plt.plot(l_x, l_y2, linestyle='dashed', color='blue')

    #plot mean error over iterations
    e_x = np.linspace(1, n_epochs, n_epochs)
    plt.subplot(122)
    plt.title('Root mean squared error')
    plt.plot(e_x, rmse)
    plt.show()
