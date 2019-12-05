import numpy as np

class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y, self.t = None, None
        self.delta = 1e-7

    def softmax(self, x):
        if x.ndim == 2:
            x -= x.max(axis=1, keepdims=True)
            x = np.exp(x)
            x /= x.sum(axis=1, keepdims=True)
        elif x.ndim == 1:
            x -= np.max(x)
            x = np.exp(x)
            x /= np.sum(x)
        return x

    def cross_entropy_error(self, y, t):
        if y.ndim == 1:
            y = y.reshape(1, y.size)
        if t.ndim == 1:
            t = t.reshape(1, t.size)
        if t.size == y.size:
            t = t.argmax(axis=1)
        return -np.sum(np.log(y[np.arange(y.shape[0]), t] + self.delta)) / y.shape[0]

    def forward(self, x, t):
        self.t = t
        self.y = self.softmax(x)
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)
        return self.cross_entropy_error(self.y, self.t)

    def backward(self, dout=1):
        size = self.t.shape[0]
        if self.t.size == self.y.size:
            return (self.y - self.t) / size
        dx = self.y.copy()
        dx[np.arange(size), self.t] -= 1
        return dx / size
