import numpy as np

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def d_mse(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

def binary_crossentropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def d_binary_crossentropy(y_true, y_pred):
    return (y_pred - y_true) / (y_pred * (1 - y_pred))
