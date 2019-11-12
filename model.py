from dropout import Dropout
from batchNormalization import BatchNormalization

class Model:
    def __init__(self):
        self.params, self.grads, self.layers = [], [], []
        self.loss = None

    def append(self, layer):
        self.layers.append(layer)
        self.params += layer.params
        self.grads += layer.grads

    def append_loss(self, layer):
        self.loss = layer

    def predict(self, x, train=False):
        for layer in self.layers:
            if isinstance(layer, Dropout) or isinstance(layer, BatchNormalization):
                x = layer.forward(x, train)
            else:
                x = layer.forward(x)
        return x

    def forward(self, x, t, train=True):
        y = self.predict(x, train)
        return self.loss.forward(y, t)

    def backward(self, dout=1):
        dout = self.loss.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def summary(self):
        print('-' * 50)
        for layer in self.layers:
            print(type(layer))
            for param in layer.params:
                print(param.shape)
        print(type(self.loss))
        print('-' * 50)
