# -*- coding: utf-8 -*-

import os
import requests
import gzip
import struct
import numpy as np
import array
import matplotlib.pyplot as plt


FILES = {
    'train_images': 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
    'train_labels': 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
    'test_images': 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
    'test_labels': 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
}

DTYPES = {
    8: np.uint8,
    9: np.int8,
    11: np.int16,
    12: np.int32,
    13: np.float32,
    14: np.double,
}

MAIN_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

def create_dir(dirname):
    """
    Create directory if not exists.

    Parameters
    ----------
    dirname : str
        Name of directory.

    Returns
    -------
    None.

    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)

create_dir(MAIN_DIR)


def download_data(key):
    """
    Download

    Parameters
    ----------
    key : { 'train_images', 'train_labels', 'test_images', 'test_labels' }
        One of FILES dictionary key.

    Returns
    -------
    None.

    """
    file_path = os.path.join(MAIN_DIR, key)
    if os.path.exists(file_path):
        return
    req = requests.get(FILES[key])
    content = gzip.decompress(req.content)
    with open(file_path, 'wb') as f:
        f.write(content)


def parse_data(key):
    """
    Parse donwloaded mnist dataset.

    Parameters
    ----------
    key : { 'train_images', 'train_labels', 'test_images', 'test_labels' }
        Downloaded filename.

    Returns
    -------
    ndarray
        Parsed dataset.

    """
    file_path = os.path.join(MAIN_DIR, key)
    with open(file_path, 'rb') as f:
        data = f.read()
    _, dtype, dim = struct.unpack('>HBB', data[:4])
    shape = struct.unpack('>'+'I'*dim, data[4:4*(dim+1)])
    return np.ndarray(buffer=array.array('B', data[4*(dim+1):]), dtype=DTYPES[dtype], shape=shape)


def load_mnist():
    """
    Load MNIST data.

    Returns
    -------
    tuple (train_images, train_labels, test_images, test_labels)
        MNIST data.

    """
    data = {}
    for key in FILES:
        download_data(key)
        data[key] = parse_data(key)
    return data['train_images'], data['train_labels'], data['test_images'], data['test_labels']


def show_image(image_matrix):
    """
    Plot image.

    Parameters
    ----------
    image_matrix :  ndarray
        Image to plot.

    Returns
    -------
    None.

    """
    plt.figure(1, figsize=(3, 3))
    plt.matshow(image_matrix, cmap=plt.cm.gray)
    plt.show()
    
    
def show_image_pixels(image_matrix):
    """
    Plot image and pixels.

    Parameters
    ----------
    image_matrix : ndarray
        Image to plot.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image_matrix, cmap=plt.get_cmap('YlGnBu'))
    row, cols = image_matrix.shape
    ax.set_xticks(np.arange(row))
    ax.set_yticks(np.arange(cols))
    for i in range(row):
        for j in range(cols):
            ax.text(j, i, image_matrix[i, j], ha='center', va='center', color='w' if image_matrix[i, j] > 128 else 'k')
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    train_images, train_labels, test_images, test_labels = load_mnist()
    print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)
    rand_index = np.random.randint(0, train_images.shape[0])
    show_image_pixels(train_images[rand_index])
