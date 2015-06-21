# this file defines function for parsing the MNIST dataset as available to
# download here:
# http://yann.lecun.com/exdb/mnist/index.html
# The extracted files are expected to reside in a subfolder called Data/MNIST/

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os

path = "Data/MNIST/"

#reads training and test labels as numpy.arrays
def read_labels():
    training_labels = read_labels_from_file(path + "train-labels.idx1-ubyte")
    test_labels = read_labels_from_file(path + "t10k-labels.idx1-ubyte")
    return training_labels, test_labels
    
def read_labels_from_file(filepath):
    f = open(filepath, "rb")
    magic = np.fromfile(f, dtype='>i4', count=1)
    n_labels = np.fromfile(f, dtype='>i4', count=1)
    labels = np.fromfile(f, dtype='i1')
    return labels

#reads training and test images as numpy.arrays
def read_images():
    training_images = read_images_from_file(path + "train-images.idx3-ubyte")
    test_images = read_images_from_file(path + "t10k-images.idx3-ubyte")
    return training_images, test_images
    
def read_images_from_file(filepath):
    f = open(filepath, "rb")
    magic = np.fromfile(f, dtype='>i4', count=1)
    n_images = np.fromfile(f, dtype='>i4', count=1)
    n_rows = np.fromfile(f, dtype='>i4', count=1)
    n_columns = np.fromfile(f, dtype='>i4', count=1)
    pixels = np.fromfile(f, dtype='B').reshape(n_images, n_rows*n_columns)
    return pixels

#shows all images in sequence
def show_test_images():
    tr, te = read_images()
    show_images(te, "MNIST test images")
    
def show_training_images():
    tr, re = read_images()
    show_images(tr, "MNIST training images")
        
def show_images(images, title):
    im = plt.imshow(images[0].reshape(28,28), cmap=plt.cm.gray, interpolation='nearest')
    plt.title(title, fontsize=20)
    for i in range(images.shape[0]):
        im.set_data(images[i].reshape(28,28))
        plt.pause(.1)
        plt.draw()
        
