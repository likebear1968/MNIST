import numpy as np

def scale(count, typ=0):
    if typ == 0:
        return np.sqrt(2.0 / count) #He
    return np.sqrt(1.0 / count) # Xavier

class CNNUtil:
    def __init__(self, fh, fw, stride=1, pad=0):
        self.fh, self.fw, self.stride, self.pad = fh, fw, stride, pad
        self.N, self.C, self.H, self.W = None, None, None, None
        self.oh, self.ow = None, None

    def im2col(self, im):
        self.N, self.C, self.H, self.W = im.shape
        self.oh = (self.H + 2 * self.pad - self.fh) // self.stride + 1
        self.ow = (self.W + 2 * self.pad - self.fw) // self.stride + 1
        img = np.pad(im, [(0,0), (0,0), (self.pad, self.pad), (self.pad, self.pad)], 'constant')
        col = np.zeros((self.N, self.C, self.fh, self.fw, self.oh, self.ow))
        for h in range(self.fh):
            h_max = h + self.stride * self.oh
            for w in range(self.fw):
                w_max = w + self.stride * self.ow
                col[:, :, h, w, :, :] = img[:, :, h: h_max: self.stride, w: w_max: self.stride]
        return col.transpose(0, 4, 5, 1, 2, 3).reshape(self.N * self.oh * self.ow, -1)

    def col2im(self, col):
        col = col.reshape(self.N, self.oh, self.ow, self.C, self.fh, self.fw).transpose(0, 3, 4, 5, 1, 2)
        img = np.zeros((self.N, self.C, self.H + 2 * self.pad + self.stride - 1, self.W + 2 * self.pad + self.stride - 1))
        for h in range(self.fh):
            h_max = h + self.stride * self.oh
            for w in range(self.fw):
                w_max = w + self.stride * self.ow
                img[:, :, h: h_max: self.stride, w: w_max: self.stride] += col[:, :, h, w, :, :]
        return img[:, :, self.pad: self.H + self.pad, self.pad: self.W + self.pad]
