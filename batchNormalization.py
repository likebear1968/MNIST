import numpy as np

class BatchNormalization:
    def __init__(self, gamma, beta, momentum=0.9, mean=None, var=None):
        self.params, self.grads = [], []
        self.gamma, self.beta, self.momentum, self.mean, self.var = gamma, beta, momentum, mean, var
        self.input_shape, self.xc, self.std, self.dgamma, self.dbeta = None, None, None, None, None
        self.delta = 1e-6

    def __forward(self, x, train):
        if self.mean is None:
            _, D = x.shape
            self.mean, self.var = np.zeros(D), np.zeros(D)
        if train:
            mu = x.mean(axis=0)
            self.xc = x - mu
            var = np.mean(self.xc ** 2, axis=0)
            self.std = np.sqrt(var + self.delta)
            self.xn = xn = self.xc / self.std
            self.mean = self.momentum * self.mean + (1 - self.momentum) * mu
            self.var = self.momentum * self.var + (1 - self.momentum) * var
        else:
            xn = (x - self.mean) / np.sqrt(self.var + self.delta)
        return self.gamma * xn + self.beta

    def forward(self, x, train=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, _, _, _ = x.shape
            x = x.reshape(N, -1)
        out = self.__forward(x, train)
        return out.reshape(*x.shape)

    def __backward(self, dout):
        self.dbeta = dout.sum(axis=0)
        self.dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std **2), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.input_shape[0]) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        return dxc - dmu / self.input_shape[0]

    def backward(self, dout):
        if dout.ndim != 2:
            N, _, _, _ = dout.shape
            dout = dout.reshape(N, -1)
        dx = self.__backward(dout)
        return dx.reshape(*self.input_shape)
