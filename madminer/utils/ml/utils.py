from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import torch


def get_activation(activation):
    if activation == 'relu':
        return torch.relu
    elif activation == 'tanh':
        return torch.tanh
    elif activation == 'sigmoid':
        return torch.sigmoid
    else:
        raise ValueError('Activation function %s unknown', activation)


def s_from_r(r):
    return np.clip(1. / (1. + r), 0., 1.)


def r_from_s(s, epsilon=1.e-6):
    return np.clip((1. - s + epsilon) / (s + epsilon), epsilon, None)


def sigmoid(x):
    return 1. / (1. + np.exp(-x))
