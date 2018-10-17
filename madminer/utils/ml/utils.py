from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import torch


def get_activation_function(activation):
    """

    Parameters
    ----------
    activation :
        

    Returns
    -------

    """
    if activation == 'relu':
        return torch.relu
    elif activation == 'tanh':
        return torch.tanh
    elif activation == 'sigmoid':
        return torch.sigmoid
    else:
        raise ValueError('Activation function %s unknown', activation)


def s_from_r(r):
    """

    Parameters
    ----------
    r :
        

    Returns
    -------

    """
    return np.clip(1. / (1. + r), 0., 1.)


def r_from_s(s, epsilon=1.e-6):
    """

    Parameters
    ----------
    s :
        
    epsilon :
         (Default value = 1.e-6)

    Returns
    -------

    """
    return np.clip((1. - s + epsilon) / (s + epsilon), epsilon, None)


def sigmoid(x):
    """

    Parameters
    ----------
    x :
        

    Returns
    -------

    """
    return 1. / (1. + np.exp(-x))


def check_for_nans_in_parameters(model, check_gradients=True):
    """

    Parameters
    ----------
    model :
        
    check_gradients :
         (Default value = True)

    Returns
    -------

    """
    for param in model.parameters():
        if torch.any(torch.isnan(param)):
            return True

        if check_gradients and torch.any(torch.isnan(param.grad)):
            return True

    return False
