from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


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


def check_required_data(method, r_xz, t_xz0, t_xz1, theta0, theta1, x, y):
    data_is_there = True
    if x is None:
        data_is_there = False
    if (
        method
        in [
            "carl",
            "carl2",
            "nde",
            "scandal",
            "rolr",
            "alice",
            "rascal",
            "alices",
            "rolr2",
            "alice2",
            "rascal2",
            "alices2",
        ]
        and theta0 is None
    ):
        data_is_there = False
    if method in ["rolr", "alice", "rascal", "alices", "rolr2", "alice2", "rascal2", "alices2"] and r_xz is None:
        data_is_there = False
    if (
        method in ["carl", "carl2", "rolr", "alice", "rascal", "alices", "rolr2", "alice2", "rascal2", "alices2"]
        and y is None
    ):
        data_is_there = False
    if method in ["scandal", "rascal", "alices", "rascal2", "alices2", "sally", "sallino"] and t_xz0 is None:
        data_is_there = False
    if method in ["carl2", "rolr2", "alice2", "rascal2", "alices2"] and theta1 is None:
        data_is_there = False
    if method in ["rascal2", "alices2"] and t_xz1 is None:
        data_is_there = False
    return data_is_there
