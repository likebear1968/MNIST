import numpy as np
from util import CNNUtil

class Pooling:
    def __init__(self, ph=2, pw=2, stride=2, pad=0):
        self.params, self.grads = [], []
        self.ph, self.pw, self.stride, self.pad = ph, pw, stride, pad
        self.argmax = None
        self.u = CNNUtil(ph, pw, stride, pad)

    def forward(self, x):
        N, C, H, W = x.shape
        col = self.u.im2col(x).reshape(-1, self.ph * self.pw)
        self.argmax = np.argmax(col, axis=1)
        oh = (H - self.ph) // self.stride + 1
        ow = (W - self.pw) // self.stride + 1
        return np.max(col, axis=1).reshape(N, oh, ow, C).transpose(0, 3, 1, 2)

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        size = self.ph * self.pw
        dmax = np.zeros((dout.size, size))
        dmax[np.arange(self.argmax.size), self.argmax.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (size,))
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        return self.u.col2im(dcol)
