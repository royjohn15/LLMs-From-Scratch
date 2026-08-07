import numpy as np

class Activation:
    def forward(self, x):
        raise NotImplementedError
    def backward(self, grad_output):
        raise NotImplementedError

class ReLU(Activation):
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)
    def backward(self, grad_output):
        grad = grad_output.copy()
        grad[self.input <= 0] = 0 # gradient is 0 for negative values and 1 for positive values
        return grad

class Sigmoid(Activation):
    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output
    def backward(self, grad_output):
        grad = grad_output.copy()
        grad = grad * self.output * (1 - self.output)
        return grad

class Tanh(Activation):
    def forward(self, x):
        self.output = np.tanh(x)
        return self.output
    def backward(self, grad_output):
        grad = grad_output.copy()
        grad = grad * (1 - self.output**2)
        return grad

class Softmax(Activation):
    def forward(self, x):
        self.output = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        return self.output
    def backward(self, grad_output):
        grad = grad_output.copy()
        return grad
