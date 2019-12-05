import numpy as np

class SigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.loss, self.y, self.t = None, None, None
        self.delta = 1e-7

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def binary_cross_entropy_error(self, y, t):
        if y.ndim == 1:
            y = y.reshape(1, y.size)
        if t.ndim == 1:
            t = t.reshape(1, t.size)
        return -np.sum(t * np.log(y + self.delta) + (1 - t) * np.log(1 - y + self.delta)) / y.shape[0]

    def forward(self, x, t):
        self.t = t.reshape(-1, 1)
        self.y = self.sigmoid(x)
        self.loss = self.binary_cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        return (self.y - self.t) * dout / self.t.shape[0]
