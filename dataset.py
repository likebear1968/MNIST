import os.path
import urllib.request
import gzip
import numpy as np

url = 'http://yann.lecun.com/exdb/mnist/'
file_names = [
    'train-images-idx3-ubyte.gz',
    'train-labels-idx1-ubyte.gz',
    't10k-images-idx3-ubyte.gz',
    't10k-labels-idx1-ubyte.gz'
]

def download(url, file_from, file_to):
    if not os.path.exists(file_to):
        urllib.request.urlretrieve(url + file_from, file_to)
    return file_to

def unpack(file_name, offset):
    with gzip.open(file_name, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=offset)
    return data

def get_images(url, file_name, size=784, normalize=True, shape=(1, 28, 28)):
    local = download(url, file_name, './' + file_name)
    data = unpack(local, 16).reshape(-1, size)
    if normalize:
        data = data.astype(np.float32) / 255.0
    if shape is not None:
        data = data.reshape(-1, *shape)
    return data

def get_labels(url, file_name, one_hot=10):
    local = download(url, file_name, './' + file_name)
    data = unpack(local, 8)
    if one_hot is not None:
        T = np.zeros((data.size, one_hot))
        for i, t in enumerate(data):
            T[i] = [1 if t == i else 0 for i in range(10)]
        return np.array(T)
    return np.array(data)
