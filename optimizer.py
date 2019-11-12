import numpy as np

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]

class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr, self.momentum = lr, momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = [np.zeros_like(p) for p in params]
        for i in range(len(params)):
            self.v[i] = self.momentum * self.v[i] - self.lr * grads[i]
            params[i] += self.v[i]

class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.v = None
        self.eta = 1e-7

    def update(self, params, grads):
        if self.v is None:
            self.v = [np.zeros_like(p) for p in params]
        for i in range(len(params)):
            self.v[i] += grads[i] * grads[i]
            params[i] -= self.lr * grads[i] / (np.sqrt(self.v[i]) + self.eta)

class RMSprop:
    def __init__(self, lr=0.01, decay_rate = 0.99):
        self.lr, self.decay_rate = lr, decay_rate
        self.v = None
        self.eta = 1e-7

    def update(self, params, grads):
        if self.v is None:
            self.v = [np.zeros_like(p) for p in params]
        for i in range(len(params)):
            self.v[i] *= self.decay_rate
            self.v[i] += (1 - self.decay_rate) * grads[i] * grads[i]
            params[i] -= self.lr * grads[i] / (np.sqrt(self.v[i]) + self.eta)

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr, self.beta1, self.beta2 = lr, beta1, beta2
        self.m, self.v = None, None
        self.iter = 0
        self.eta = 1e-7

    def update(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
        if self.v is None:
            self.v = [np.zeros_like(p) for p in params]
        self.iter += 1
        lr_t = self.lr * np.sqrt(1 - self.beta2 ** self.iter) / (1 - self.beta1 ** self.iter)
        for i in range(len(params)):
            self.m[i] += (1 - self.beta1) * (grads[i] - self.m[i])
            self.v[i] += (1 - self.beta2) * (grads[i] ** 2 - self.v[i])
            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + self.eta)
