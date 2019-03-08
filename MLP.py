#!/usr/bin/env python
import numpy as np


def sigmoid(x):
    """ Sigmoid like function using tanh """
    return np.tanh(x)


def dsigmoid(x):
    """ Derivative of sigmoid above """
    return 1.0-x**2


class MLP:
    """ Multi-layer perceptron class. """

    def __init__(self, *args):
        """ Initialization of the perceptron with given sizes.  """

        self.shape = args
        n = len(args)
        # print "n :",n

        # Build layers
        self.layers = []
        # Input layer (+1 unit for bias)
        #self.layers.append(np.ones(self.shape[0]+1))
        self.layers.append(np.ones(self.shape[0]))
        # Hidden layer(s) + output layer
        for i in range(1, n):
            self.layers.append(np.ones(self.shape[i]))
            # print 'layers', self.layers
            # print "i", i

        # Build weights matrix (randomly between -0.25 and +0.25)
        self.weights = []
        for i in range(n-1):
            self.weights.append(np.zeros((self.layers[i].size,
                                         self.layers[i+1].size)))

        # dw will hold last change in weights (for momentum)
        self.dw = [0] * len(self.weights)

        # Reset weights
        self.reset()

    def reset(self):
        """ Reset weights """
        for i in range(len(self.weights)):
            self.weights[i] = np.random.uniform(-0.25, 0.25,
                                                (len(self.weights[i]),
                                                 len(self.weights[i][0])))

    def propagate_forward(self, data):
        """ Propagate data from input layer to output layer. """
        print(data)
        print(self.layers[0])
        for d in range(len(data)):
            self.layers[0][d] = data[d]
        for i in range(len(self.weights)):
            self.layers[i + 1] = sigmoid(np.dot(self.layers[i], self.weights[i]))


    def propagate_backward(self, target, lrate=0.001, momentum=0.1):
        """ Back propagate error related to target using lrate. """
