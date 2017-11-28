# coding:utf-8
'''
Created on 2017/11/28.

@author: chk01
'''
import os
import numpy as np
import tensorflow as tf

data_dir = 'data/mnist'


def load_mnist(type='train'):
    if type == 'train':
        fd = open(os.path.join(data_dir, 'train-images.idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        X = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)
        res_X = tf.convert_to_tensor(X / 255., tf.float32)
        fd = open(os.path.join(data_dir, 'train-labels.idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        res_Y = loaded[8:].reshape((60000, 1)).astype(np.int32)
    else:
        fd = open(os.path.join(data_dir, 't10k-images.idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        X = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)
        res_X = X / 255.
        fd = open(os.path.join(data_dir, 't10k-labels.idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        res_Y = loaded[8:].reshape((10000, 1)).astype(np.int32)

    return res_X, res_Y

# x, y = load_mnist('train')
