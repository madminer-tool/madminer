from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import torch
import logging
from torch import optim

import madminer.utils
from madminer.utils.ml import losses

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


def get_optimizer(optimizer, nesterov_momentum):
    opt_kwargs = None
    if optimizer == "adam":
        opt = optim.Adam
    elif optimizer == "amsgrad":
        opt = optim.Adam
        opt_kwargs = {"amsgrad": True}
    elif optimizer == "sgd":
        opt = optim.SGD
        if nesterov_momentum is not None:
            opt_kwargs = {"momentum": nesterov_momentum}
    else:
        raise ValueError("Unknown optimizer {}".format(optimizer))
    return opt, opt_kwargs


def get_loss(method, alpha):
    if method in ["carl", "carl2"]:
        loss_functions = [losses.ratio_xe]
        loss_weights = [1.0]
        loss_labels = ["xe"]
    elif method in ["rolr", "rolr2"]:
        loss_functions = [losses.ratio_mse]
        loss_weights = [1.0]
        loss_labels = ["mse_r"]
    elif method == "cascal":
        loss_functions = [losses.ratio_xe, losses.ratio_score_mse_num]
        loss_weights = [1.0, alpha]
        loss_labels = ["xe", "mse_score"]
    elif method == "cascal2":
        loss_functions = [losses.ratio_xe, losses.ratio_score_mse]
        loss_weights = [1.0, alpha]
        loss_labels = ["xe", "mse_score"]
    elif method == "rascal":
        loss_functions = [losses.ratio_mse, losses.ratio_score_mse_num]
        loss_weights = [1.0, alpha]
        loss_labels = ["mse_r", "mse_score"]
    elif method == "rascal2":
        loss_functions = [losses.ratio_mse, losses.ratio_score_mse]
        loss_weights = [1.0, alpha]
        loss_labels = ["mse_r", "mse_score"]
    elif method in ["alice", "alice2"]:
        loss_functions = [losses.ratio_augmented_xe]
        loss_weights = [1.0]
        loss_labels = ["improved_xe"]
    elif method == "alices":
        loss_functions = [losses.ratio_augmented_xe, losses.ratio_score_mse_num]
        loss_weights = [1.0, alpha]
        loss_labels = ["improved_xe", "mse_score"]
    elif method == "alices2":
        loss_functions = [losses.ratio_augmented_xe, losses.ratio_score_mse]
        loss_weights = [1.0, alpha]
        loss_labels = ["improved_xe", "mse_score"]
    elif method in ["sally", "sallino"]:
        loss_functions = [losses.local_score_mse]
        loss_weights = [1.0]
        loss_labels = ["mse_score"]
    elif method == "nde":
        loss_functions = [madminer.utils.ml.losses.flow_nll]
        loss_weights = [1.0]
        loss_labels = ["nll"]
    elif method == "scandal":
        loss_functions = [madminer.utils.ml.losses.flow_nll, madminer.utils.ml.losses.flow_score_mse]
        loss_weights = [1.0, alpha]
        loss_labels = ["nll", "mse_score"]
    else:
        raise NotImplementedError("Unknown method {}".format(method))
    return loss_functions, loss_labels, loss_weights
