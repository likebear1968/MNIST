import numpy as np
from util import CNNUtil

class Convolution:
    def __init__(self, w, b, stride=1, pad=0):
        self.params = [w, b]
        self.grads = [np.zeros_like(w), np.zeros_like(b)]
        self.col, self.col_w = None, None
        _, _, FH, FW = w.shape
        self.u = CNNUtil(FH, FW, stride, pad)

    def forward(self, x):
        w, b = self.params
        FN, C, FH, FW = w.shape
        N, _, H, W = x.shape
        self.col = self.u.im2col(x)
        self.col_w = w.reshape(FN, -1)
        out = np.dot(self.col, self.col_w.T) + b
        return out.reshape(N, self.u.oh, self.u.ow, -1).transpose(0, 3, 1, 2)

    def backward(self, dout):
        w, _ = self.params
        FN, C, FH, FW = w.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)
        dw = np.dot(self.col.T, dout)
        self.grads[0][...] = dw.transpose(1, 0).reshape(FN, C, FH, FW)
        self.grads[1][...] = np.sum(dout, axis=0)
        return  self.u.col2im(np.dot(dout, self.col_w))
