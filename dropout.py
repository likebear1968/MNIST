import numpy as np

class Dropout:
    def __init__(self, ratio=0.5):
        self.ratio = ratio
        self.mask = None
        self.params, self.grads = [], []

    def forward(self, x, train=True):
        if train:
            self.mask = np.random.rand(*x.shape) > self.ratio
            return x * self.mask
        self.mask = np.ones_like(*x.shape, dtype=bool)
        return x * (1.0 - self.ratio)

    def backward(self, dout):
        return dout * self.mask
