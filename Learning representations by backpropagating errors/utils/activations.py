import numpy as np

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

# Derivative of activation functions
def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def d_tanh(x):
    return 1 - np.tanh(x)**2

def d_relu(x):
    return np.where(x > 0, 1, 0)
