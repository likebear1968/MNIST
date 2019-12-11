import numpy as np

class IdentityWithLoss():
    def __init__(self):
        self.params, self.grads = [], []
        self.loss, self.y, self.t = None, None, None

    def mse(self, y, t):
        return np.average((t - y) ** 2, axis=0)

    def forward(self, x, t):
        self.t = t.reshape(-1, 1)
        self.y = x
        self.loss = self.mse(self.y, self.t)
        return self.loss, self.y

    def backward(self, dout=1):
        return (self.y - self.t) * dout / self.t.shape[0]