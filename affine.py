import numpy as np

class Affine:
    def __init__(self, w, b):
        self.params = [w, b]
        self.grads = [np.zeros_like(w), np.zeros_like(b)]
        self.x, self.x_shape = None, None

    def forward(self, x):
        self.x_shape = x.shape
        self.x = x.reshape(x.shape[0], -1)
        return np.dot(self.x, self.params[0]) + self.params[1]

    def backward(self, dout):
        self.grads[0][...] = np.dot(self.x.T, dout)
        self.grads[1][...] = np.sum(dout, axis=0)
        dx = np.dot(dout, self.params[0].T)
        return dx.reshape(*self.x_shape)
