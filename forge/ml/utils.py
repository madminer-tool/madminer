import numpy as np
import os
import logging
import torch.nn.functional as F


def get_activation(activation):
    if activation == 'relu':
        return F.relu
    elif activation == 'tanh':
        return F.tanh
    elif activation == 'sigmoid':
        return F.sigmoid
    else:
        raise ValueError('Activation function %s unknown', activation)
