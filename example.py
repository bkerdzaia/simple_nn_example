#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from nn import Layer, NN, relu, softmax
from mnist import load_mnist


def one_hot_vector(x, size):
    """
    Convert labels to one-hot vector.

    Parameters
    ----------
    x : ndarray
        array of labels.
    size : int
        total number of unique labels.

    Returns
    -------
    ndarray
        one-hot vector.

    """
    return np.eye(size)[x, :]


def normalize(images):
    """
    Normalize image pixels by dividing 255.

    Parameters
    ----------
    images : ndarray
        Array of image pixels.

    Returns
    -------
    X : ndarray
        Normalized image.

    """
    X = images.reshape(images.shape[0], -1)
    X = X / 255
    return X


def main():
    train_images, train_labels, test_images, test_labels = load_mnist()
    X = normalize(train_images)
    label_size = len(np.unique(train_labels))
    y = one_hot_vector(train_labels, label_size)

    print("Total training example:", X.shape[0])

    nn = NN(epoch=20, batch_size=256)

    nn.add_layer(Layer(784))
    nn.add_layer(Layer(200, activation_fn=relu))
    nn.add_layer(Layer(100, activation_fn=relu))
    nn.add_layer(Layer(10, activation_fn=softmax))

    nn.fit(X, y)


    print("Train Accuracy is:", nn.accuracy(X, y))

    X_test = normalize(test_images)
    Y_test = one_hot_vector(test_labels, label_size)
    print("Test Accuracy is:", nn.accuracy(X_test, Y_test))

    nn.plot_learning_curve()


if __name__ == '__main__':
    main()
