# short lectures on backpropagation and softmax by Geoffrey Hinton:
#
# Backpropagation: https://www.youtube.com/watch?v=H47Y7pAssTI
#
# Softmax: https://www.youtube.com/watch?v=yqsI-X40OBY
#
# This example is very similar to the BackPropagation example. The key difference
# is that the last layer is a softmax with two units instead of a single 
# logistic unit. We train the softmax to represent a probability distribution 
# over two classes ('pos' and 'neg'). To do that, we need to change the labels
# from scalar ('1' for examples from the 'pos' class and 0 for 'neg' samples) to
# vector representation encoding class membership for each class individually.
# The switch to softmax also allows us to minimize cross-entropy error instead
# of least squares, which is a better fit for classification problems as explained
# in the softmax-video linked above.

import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.insert(0, "../..")
sys.path.insert(0, "..")
import PyRat as pr

def run():
    #generate n samples of random 2d data from each class
    #such that classes are not linearly separable
    n = 400
    m_pos1 = np.array([-0.5, -0.5]).reshape(2, -1)
    x_pos1 = np.random.randn(2, n/2)/4 + m_pos1
    m_pos2 = np.array([0.5, 0.5]).reshape(2, -1)
    x_pos2 = np.random.randn(2, n/2)/4 + m_pos2
    x_pos = np.concatenate((x_pos1, x_pos2), axis=1)
    ones = np.ones(n)
    zeros = np.zeros(n)
    y_pos = np.vstack((ones, zeros))
    
    m_neg1 = np.array([-0.5, 0.5]).reshape(2, -1)    
    x_neg1 = np.random.randn(2, n/2)/4 + m_neg1
    m_neg2 = np.array([0.5, -0.5]).reshape(2, -1)    
    x_neg2 = np.random.randn(2, n/2)/4 + m_neg2
    x_neg = np.concatenate((x_neg1, x_neg2), axis=1)
    y_neg = np.vstack((zeros, ones))

    #concatenate positive and negative examples
    x = np.concatenate((x_pos, x_neg), axis=1)
    y = np.concatenate((y_pos, y_neg))
    
    #set y as 't'arget
    t = y.reshape(2, -1)
    
    #initialize hidden layer with 2 logistic units
    n_dims = len(x)
    n0 = 4
    h0 = pr.logistic(n0, n_dims)
    #initialize output layer with 1 logistic unit
    h1 = pr.softmax(2, n0)
    
    #define number of training epochs (decreasing learning rate)
    n_epochs = 1
    #number of iterations in each epoch
    n_iter = 5000
    #learning rate
    learning_rate = 1.
    
    cost = np.zeros(n_epochs*n_iter)
    
    #train network
    for e in range(n_epochs):
        for i in range(n_iter):
            y0 = h0.forward(x)
            cost[e*n_iter + i], dEdx1 = h1.evaluate(y0, t, learning_rate)
            h0.backprop(dEdx1, learning_rate)
            
            #print error and weights at end of each iteration
            if(i % 1000 == 0):
                print "Cost:" + str(cost[e*n_iter+i]) + " w: " + str(h0.w)
        learning_rate *= 0.1
        
    #plot data
    plt.figure(figsize=(20, 9))
    plt.subplot(121)
    plt.title('Separation in feature space')
    plt.scatter(x_pos[0,:], x_pos[1,:], color='blue', marker='+')
    plt.scatter(x_neg[0,:], x_neg[1,:], color='red', marker='o')
    
    #separation line at p=0.5 is defined as x2 = -w1/w2 * x1 - w0/w2
    w = h0.w
    l_x = np.linspace(-1, 1, 2)
    for i in range(h0.n_outputs):
        l_y = -w[0,i]/w[1,i] * l_x -w[n_dims,i]/w[1,i]
        plt.plot(l_x, l_y, color='black')
    
    #plot mean error over iterations
    e_x = np.linspace(1, n_epochs*n_iter, n_epochs*n_iter)
    plt.subplot(122)
    plt.title('Cross-entropy error')
    plt.plot(e_x, cost)
    plt.show()
