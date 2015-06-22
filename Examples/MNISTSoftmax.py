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
    
    #labels are vectors with numbers in range [0,9], whereas the softmax layer
    #expects labels to be indicator functions over vectors with one element per
    #class.
    def vectorize_labels(labels, n_classes):
        l = np.zeros((len(labels), n_classes)) #get all zero array of target size
        a = np.arange(len(labels)) #column index for each label
        l[a, labels] = 1 #set in each section of 10 the correct index to 1
        return l
    labels_train = vectorize_labels(labels_train, 10)
    
    #separate training labels and training data into batches of fix size
    batch_size = 100
    n_batches = data_train.shape[0]/batch_size
    batch_data = data_train.reshape(-1, batch_size, data_train.shape[1])
    batch_labels = labels_train.reshape(-1, batch_size, labels_train.shape[1])
    
    #Transpose data, because data rows represent images and columns pixels, 
    #neuron layers expect rows to be features (pixels) and columns to represent
    # samples (images).
    batch_data = batch_data.transpose(0, 2, 1)
    batch_labels = batch_labels.transpose(0, 2, 1)
    data_test = data_test.T
    
    #data is in range [0, 255], but we need it to be within [-0.5,0.5]
    batch_data = batch_data/255 - 0.5
    data_test = data_test/255 - 0.5
    
    #initialize network: here we use a single softmax layer with one unit per class
    n_features = batch_data.shape[1]
    h0 = pr.softmax(10, n_features)
    
    #number of training epochs (learning rate decreases after each epoch)
    n_epochs = 3
    #number of iterations per epoch (all data is seen once per iteration)
    n_iter = 10000
    #learning rate
    learning_rate = 0.01
    #track cross-entropy error as it decreases over time (plotted after training)
    cost = np.zeros(n_epochs*n_iter*n_batches)
    
    #train network
    start = time.time()
    for e in range(n_epochs):
    
        for i in range(n_iter):
            batch_cost = 0
            for b in range(n_batches):
                #pass training data through softmax, estimate error and backpropagate
                C, dEdx1 = h0.evaluate(batch_data[b], batch_labels[b], learning_rate)
                batch_cost += C
                
            cost[e*n_iter + i] = batch_cost/n_batches
            #print error in each iteration
            print "error:" + str(cost[e*n_iter + i])    
            #estimate error on test set every n iterations
            if(i % 10 == 0):
                #feed test set through the network
                prediction = h0.forward(data_test)
                #find for each sample the index in label vector with highest probability
                l_p = np.argmax(prediction, axis=0)
                #compare predicted labels to ground truth
                diff = labels_test - l_p
                #count number of false predictions
                errors = np.count_nonzero(diff)
                
                print str(e*n_iter + i) + ": test error: " + str(errors) +" - " + str((errors*100.0)/len(labels_test)) + "%"
                
        #adapt learning at end of each epoch
        learning_rate *= 0.1
    
    #measure training time
    end = time.time()
    print "Training time in seconds: " + str(end-start) + " - " + + str((end-start)/(n_epochs*n_iter)) + " per iteration"
    
    #plot mean error over iterations
    e_x = np.linspace(1, n_epochs*n_iter, n_epochs*n_iter)
    plt.title('Cross-entropy error')
    plt.plot(e_x, cost)
    plt.show()
