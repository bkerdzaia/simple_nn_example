#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad

import matplotlib.pyplot as plt


def softmax(x):
    """
    Computes softmax function.

    Parameters
    ----------
    x : ndarray
        input.

    Returns
    -------
    ndarray
        applied softmax function.

    """
    ex = np.exp(x - np.max(x))
    return ex / np.sum(ex, axis=0)


def relu(x):
    """
    Computes ReLU activation function.

    Parameters
    ----------
    x : ndarray
        input.

    Returns
    -------
    ndarray
        applied relu function.

    """
    return np.maximum(x, 0)


def xavier_initializer(m, n):
    """
    Computes xavier initializer[1]_.

    Parameters
    ----------
    m : int
        dimension of previous layer.
    n : int
        dimension of current layer.

    Returns
    -------
    ndarray
        intialized values by xavier initializer.
        
        
    References
    ----------
    .. [1] Glorot, Xavier & Bengio, Y.. (2010). Understanding the difficulty of training deep feedforward neural networks. http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

    """
    return npr.randn(m, n) * np.sqrt(6/(n+m))


def zero_initializer(a, b):
    """
    Return a new array filled with zeros.

    Parameters
    ----------
    a : int
        dimension of previous layer.
    b : int
        dimension of current layer.

    Returns
    -------
    ndarray
        zeros of input shape.

    """
    return np.zeros((a, b))


class Layer(object):
    """
    Layer(n_layer, activation_fn=None)
    
    Neural network layer class.

    Attributes
    ----------
    n_layer : int
        Number of layers.
    activation_fn : function, optional
        Activation function for this layer. The default is None.

    """
    def __init__(self, n_layer, activation_fn=None):
        self.n_layer = n_layer
        self.activation_fn = activation_fn if activation_fn is not None else lambda z: z


class NN(object):
    """
    NN(learning_rate=0.001, epoch=5, batch_size=1000, momentum_param=0.9, momentum_param2=0.999, epsilon=1e-8)
    
    Neural network class, that implements adaptive learning rate optimization algorithm (Adam[1]_).

    Attributes
    ----------
    learning_rate : float, optional
        Learning rate. The default is 0.001.
    epoch : int, optional
        Number of epochs. The default is 5.
    batch_size : int, optional
        Batch size. The default is 1000.
    momentum_param, momentum_param2 : float, optional
        Exponential decay rates for the moment estimates.
        momentum_param default is 0.9.
        momentum_param2 default is 0.999.
    epsilon : float, optional
        Small constant used for numerical stabilization. The default is 1e-8.

    Methods
    -------
    add_layer(layer)
        Add new layer to the network.
    fit(X, y):
        Train a model.
    predict(X):
        Predict labels.
    accuracy(X, y):
        Get accuracy of predicted values.
    plot_learning_curve():
        Plot learning curve.
        
    References
    ----------
    .. [1] Diederik, Kingma; Ba, Jimmy (2014). "Adam: A method for stochastic optimization". https://arxiv.org/abs/1412.6980v8
    
    """
    def __init__(self, learning_rate=0.001, epoch=5,
                 batch_size=1000, momentum_param=0.9,
                 momentum_param2=0.999, epsilon=1e-8):
        self.layers = []
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size
        self.momentum_param = momentum_param
        self.momentum_param2 = momentum_param2
        self.epsilon = epsilon
    
    def add_layer(self, layer):
        """
        Add layer to the network.

        Parameters
        ----------
        layer : Layer
            new layer.

        Returns
        -------
        None.

        """
        self.layers.append(layer)
    
    def _get_params(self, initializer=xavier_initializer):
        params = {}
        prev_layer = self.layers[0]
        for i in range(1, len(self.layers)):
            params['W'+str(i)] = initializer(prev_layer.n_layer, self.layers[i].n_layer)
            params['b'+str(i)] = np.zeros((1, self.layers[i].n_layer))
            prev_layer = self.layers[i]
        return params

    def _forward(self, params, X):
        # forward propagate
        A = X
        for i in range(1, len(self.layers)):
            Z = np.dot(A, params['W'+str(i)] + params['b'+str(i)])
            A = self.layers[i].activation_fn(Z)
        return Z
    
    def fit(self, X, y):
        """
        Fit the neural network model according to the given training data.

        Parameters
        ----------
        X : ndarray
            Training vectors.
        y : ndarray
            Target values.

        Raises
        ------
        Exception
            When neural network has no input and output layers or number of layers in input layer is incorrect.

        Returns
        -------
        None.

        """
        
        if len(self.layers) < 2:
            raise Exception("Neural network should have at least input layer and output layer")
        
        if X.shape[1] != self.layers[0].n_layer:
            raise Exception("Number of layers in input layer is not correct")
        
        self.costs = []
        n_sample = y.shape[0]
        Y = y
        
        total_batch_count = int(n_sample / self.batch_size) + int(n_sample % self.batch_size != 0)
        
        params = self._get_params()
        
        def cost(params, batch_from, batch_to):
            X_batch = X[batch_from:batch_to, :]
            Y_batch = Y[batch_from:batch_to, :]
            Z = self._forward(params, X_batch)
            
            A = self.layers[-1].activation_fn(Z)
            # compute cost
            logprobs = np.multiply(np.log(A), Y_batch) + np.multiply( np.log((1 - A)), (1-Y_batch) )
            cost = -1*np.sum(logprobs)
            return cost
        
        cost_gradient = grad(cost)
        
        s = self._get_params(initializer=zero_initializer) # 1st moment variable
        r = self._get_params(initializer=zero_initializer) # 2nd moment variable
        t = 0 # time step

        for k in range(0, self.epoch):
            total_cost = 0
            
            for i in range(total_batch_count):
                batch_from, batch_to = i*self.batch_size, min((i+1) * self.batch_size, n_sample)
                total_cost += cost(params, batch_from, batch_to)
                grads = cost_gradient(params, batch_from, batch_to)
                t += 1
                for key in params:
                    # Update biased first moment estimate
                    s[key] = self.momentum_param * s[key] + (1 - self.momentum_param) * grads[key]
                    # Update biased second moment estimate
                    r[key] = self.momentum_param2 * r[key] + (1 - self.momentum_param2) * (grads[key] ** 2)
                    # Correct bias
                    lr_corrected = self.learning_rate * np.sqrt(1 - self.momentum_param2 ** t) / (1 - self.momentum_param ** t)
                    delta_theta = -1 * lr_corrected * s[key] / (np.sqrt(r[key]) + self.epsilon)
                    # Apply update
                    params[key] = params[key] + delta_theta
            
            cost_avg = total_cost / n_sample
            self.costs.append(cost_avg)
            print (f"Loss after epoch {k}: {cost_avg}")
                
        self.params = params
    
    def predict(self, X):
        """
        Perform classification on samples in X.

        Parameters
        ----------
        X : ndarray
            Samples.

        Returns
        -------
        ndarray
            Class labels for samples in X.

        """
        Z = self._forward(self.params, X)
        return np.argmax(Z, axis=-1)
    
    def accuracy(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : ndarray
            Test samples.
        y : ndarray
            True labels for X.

        Returns
        -------
        float
            Mean accuracy of predicted labels and y.

        """
        y_prime = self.predict(X)
        y_real = np.argmax(y, axis=-1)
        return np.mean((y_prime == y_real))


    def plot_learning_curve(self):
        """
        Plot learning curve.

        Returns
        -------
        None.

        """
        costs = np.squeeze(self.costs)
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(self.learning_rate))
        plt.show()
