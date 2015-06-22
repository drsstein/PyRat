# This example uses the popular MNIST handwritten digit dataset, which 
# can be found on Yann LeCun's website under the following link:
# http://yann.lecun.com/exdb/mnist/index.html
# the code in this example uses ../mnist.py, which contains functions for
# reading image data and labels from the MNIST files.

import matplotlib.pyplot as plt
import numpy as np
import sys
import time

sys.path.insert(0, "../..")
sys.path.insert(0, "..")
import PyRat as pr

# The files are expected to reside in ../Data/MNIST/
pr.mnist.path = "../Data/MNIST/"

def run():
    #comment out the following line to look at the MNIST training images
    #pr.mnist.show_training_images()
    
    #read labels and data from disk
    labels_train, labels_test = pr.mnist.read_labels()
    data_train, data_test = pr.mnist.read_images()
    
    #data rows represent images and columns pixels, neuron layers expect rows to
    #be different feature dimensions (pixels) and columns to be separate images.
    data_train = data_train.T
    data_test = data_test.T
    
    #data is in range [0, 255], but we need it to be within [-0.5,0.5]
    data_train = data_train/255 - 0.5
    data_test = data_test/255 - 0.5
    
    #labels are vectors with numbers in range [0,9], whereas the softmax layer
    #expects labels to be indicator functions over vectors with one element per
    #class.
    def vectorize_labels(labels, n_classes):
        l = np.zeros((n_classes, len(labels))) #get all zero array of target size
        a = np.arange(len(labels)) #column index for each label
        l[labels, a] = 1 #set in each section of 10 the correct index to 1
        return l
    labels_train = vectorize_labels(labels_train, 10)
    
    #initialize network: here we use a single softmax layer with one unit per class
    h0 = pr.softmax(10, len(data_train))
    
    #number of training epochs (learning rate decreases after each epoch)
    n_epochs = 1
    #number of iterations per epoch (all data is seen once per iteration)
    n_iter = 10000
    #learning rate
    learning_rate = 0.05
    #track cross-entropy error as it decreases over time (plotted after training)
    cost = np.zeros(n_epochs*n_iter)
    
    #train network
    start = time.time()
    for e in range(n_epochs):
    
        for i in range(n_iter):
            cost[e*n_iter + i], dEdx1 = h0.evaluate(data_train, labels_train, learning_rate)
            print "error:" + str(cost[e*n_iter+i])
            if(i % 50 == 0):
                #evaluate test error
                #1. feed test set through the network
                prediction = h0.forward(data_test)
                l_p = np.argmax(prediction, axis=0)
                diff = labels_test - l_p
                errors = np.count_nonzero(diff)
                print str(e*n_iter + i) + ": test error: " + str(errors) +" - " + str((errors*100)/len(labels_test)) + "%"
        learning_rate *= 0.1
        
    end = time.time()
    print "Time per iteration [s]: " + str((end-start)/(n_epochs*n_iter))
    
    
    
    #plot mean error over iterations
    e_x = np.linspace(1, n_epochs*n_iter, n_epochs*n_iter)
    plt.title('Cross-entropy error')
    plt.plot(e_x, cost)
    plt.show()
