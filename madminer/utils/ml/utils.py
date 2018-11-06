from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import torch


def get_activation_function(activation):
    if activation == "relu":
        return torch.relu
    elif activation == "tanh":
        return torch.tanh
    elif activation == "sigmoid":
        return torch.sigmoid
    else:
        raise ValueError("Activation function %s unknown", activation)


def s_from_r(r):
    return np.clip(1.0 / (1.0 + r), 0.0, 1.0)


def r_from_s(s, epsilon=1.0e-6):
    return np.clip((1.0 - s + epsilon) / (s + epsilon), epsilon, None)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def check_for_nans_in_parameters(model, check_gradients=True):
    for param in model.parameters():
        if torch.any(torch.isnan(param)):
            return True

        if check_gradients and torch.any(torch.isnan(param.grad)):
            return True

    return False
