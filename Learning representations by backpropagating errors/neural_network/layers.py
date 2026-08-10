import numpy as np

class Layer:
    def forward(self, x):
        raise NotImplementedError
    def backward(self, grad_output):
        raise NotImplementedError

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.zeros((1, output_size))

    def forward(self, x):
        self.input = x
        self.output = x@self.weights + self.biases
        return self.output

    def backward(self, grad_output):
        self.d_weights = self.input.T @ grad_output
        self.d_biases = np.sum(grad_output, axis=0, keepdims=True)
        grad_input = grad_output @ self.weights.T
        return grad_input

    def update(self, learning_rate):
        self.weights -= learning_rate * self.d_weights
        self.biases -= learning_rate * self.d_biases
