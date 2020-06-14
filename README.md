# Simple Neural Network Example

Implementing Neural Network and Adam optimizer using autograd.

## Introduction

Neural network model that can dynamically add layers.

 * The weights are initialized using [Xavier](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) initializer.

 * The model is trained using [Adam](https://arxiv.org/abs/1412.6980v8) optimizer.
 
 * For backward propagation use [autograd](https://github.com/HIPS/autograd).

## How to run

The example downloads [MNIST handwritten digits database](http://yann.lecun.com/exdb/mnist/) and trains a model with one input layer, 2 hidden layers and one output layer. Hidden layers use relu activations. First hidden layer has 200 units, second - 100. The output layer uses softmax activation and 10 units, that classifies numbers 0 to 9.

```
    pip install -r requirements.txt
    python3 example.py
```
