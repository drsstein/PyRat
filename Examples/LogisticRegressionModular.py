# short lecture on learning logistic neurons by Geoffrey Hinton:
# https://www.youtube.com/watch?v=--_F0rbPH9M

import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.insert(0, "../..")
import PyRat as pr

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
    n_samples = len(x[0])
    
    #set y as 't'arget
    t = y.reshape(1, -1)
    
    #define number of training epochs (look at each example once per epoch)
    n_epochs = 200
    learning_rate = 10.
    
    #initialize layer of logistic units with a single unit
    n_dims = len(x)
    h0 = pr.logistic(1, n_dims)
    rmse = np.zeros(n_epochs)
    
    #train network
    for i in range(n_epochs):
        y = h0.forward(x)
        rmse[i], dEdy = pr.squared_error(t, y)
        h0.backprop(dEdy, learning_rate)        
        
        #print error and weights at end of each iteration
        print "RMSE:" + str(rmse[i]) + " w: " + str(h0.w)
    
    #plot data
    plt.figure(figsize=(20, 9))
    plt.subplot(121)
    plt.title('Separation in feature space')
    plt.scatter(x_pos[0,:], x_pos[1,:], color='blue', marker='+')
    plt.scatter(x_neg[0,:], x_neg[1,:], color='red', marker='o')
    
    #separation line at p=0.5 is defined as x2 = -w1/w2 * x1 - w0/w2
    w0 = h0.w[n_dims]
    w = h0.w
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
