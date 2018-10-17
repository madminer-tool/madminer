from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

import torch
from torch.nn import BCELoss, MSELoss


def ratio_mse_num(s_hat, log_r_hat, t0_hat, t1_hat, y_true, r_true, t0_true, t1_true, log_r_clip=10.):
    """

    Parameters
    ----------
    s_hat :
        
    log_r_hat :
        
    t0_hat :
        
    t1_hat :
        
    y_true :
        
    r_true :
        
    t0_true :
        
    t1_true :
        
    log_r_clip :
         (Default value = 10.)

    Returns
    -------

    """
    r_true = torch.clamp(r_true, np.exp(-log_r_clip), np.exp(log_r_clip))
    log_r_hat = torch.clamp(log_r_hat, -log_r_clip, log_r_clip)

    inverse_r_hat = torch.exp(- log_r_hat)
    return MSELoss()((1. - y_true) * inverse_r_hat, (1. - y_true) * (1. / r_true))


def ratio_mse_den(s_hat, log_r_hat, t0_hat, t1_hat, y_true, r_true, t0_true, t1_true, log_r_clip=10.):
    """

    Parameters
    ----------
    s_hat :
        
    log_r_hat :
        
    t0_hat :
        
    t1_hat :
        
    y_true :
        
    r_true :
        
    t0_true :
        
    t1_true :
        
    log_r_clip :
         (Default value = 10.)

    Returns
    -------

    """
    r_true = torch.clamp(r_true, np.exp(-log_r_clip), np.exp(log_r_clip))
    log_r_hat = torch.clamp(log_r_hat, -log_r_clip, log_r_clip)

    r_hat = torch.exp(log_r_hat)
    return MSELoss()(y_true * r_hat, y_true * r_true)


def ratio_mse(s_hat, log_r_hat, t0_hat, t1_hat, y_true, r_true, t0_true, t1_true, log_r_clip=10.):
    """

    Parameters
    ----------
    s_hat :
        
    log_r_hat :
        
    t0_hat :
        
    t1_hat :
        
    y_true :
        
    r_true :
        
    t0_true :
        
    t1_true :
        
    log_r_clip :
         (Default value = 10.)

    Returns
    -------

    """
    return (ratio_mse_num(s_hat, log_r_hat, t0_hat, t1_hat, y_true, r_true, t0_true, t1_true, log_r_clip)
            + ratio_mse_den(s_hat, log_r_hat, t0_hat, t1_hat, y_true, r_true, t0_true, t1_true, log_r_clip))


def score_mse_num(s_hat, log_r_hat, t0_hat, t1_hat, y_true, r_true, t0_true, t1_true):
    """

    Parameters
    ----------
    s_hat :
        
    log_r_hat :
        
    t0_hat :
        
    t1_hat :
        
    y_true :
        
    r_true :
        
    t0_true :
        
    t1_true :
        

    Returns
    -------

    """
    return MSELoss()((1. - y_true) * t0_hat, (1. - y_true) * t0_true)


def score_mse_den(s_hat, log_r_hat, t0_hat, t1_hat, y_true, r_true, t0_true, t1_true):
    """

    Parameters
    ----------
    s_hat :
        
    log_r_hat :
        
    t0_hat :
        
    t1_hat :
        
    y_true :
        
    r_true :
        
    t0_true :
        
    t1_true :
        

    Returns
    -------

    """
    return MSELoss()(y_true * t1_hat, y_true * t1_true)


def score_mse(s_hat, log_r_hat, t0_hat, t1_hat, y_true, r_true, t0_true, t1_true):
    """

    Parameters
    ----------
    s_hat :
        
    log_r_hat :
        
    t0_hat :
        
    t1_hat :
        
    y_true :
        
    r_true :
        
    t0_true :
        
    t1_true :
        

    Returns
    -------

    """
    return (score_mse_num(s_hat, log_r_hat, t0_hat, t1_hat, y_true, r_true, t0_true, t1_true)
            + score_mse_den(s_hat, log_r_hat, t0_hat, t1_hat, y_true, r_true, t0_true, t1_true))


def standard_cross_entropy(s_hat, log_r_hat, t0_hat, t1_hat, y_true, r_true, t0_true, t1_true):
    """

    Parameters
    ----------
    s_hat :
        
    log_r_hat :
        
    t0_hat :
        
    t1_hat :
        
    y_true :
        
    r_true :
        
    t0_true :
        
    t1_true :
        

    Returns
    -------

    """
    s_hat = 1. / (1. + torch.exp(log_r_hat))

    return BCELoss()(s_hat, y_true)


def augmented_cross_entropy(s_hat, log_r_hat, t0_hat, t1_hat, y_true, r_true, t0_true, t1_true):
    """

    Parameters
    ----------
    s_hat :
        
    log_r_hat :
        
    t0_hat :
        
    t1_hat :
        
    y_true :
        
    r_true :
        
    t0_true :
        
    t1_true :
        

    Returns
    -------

    """
    s_hat = 1. / (1. + torch.exp(log_r_hat))
    s_true = 1. / (1. + r_true)

    return BCELoss()(s_hat, s_true)


def local_score_mse(t_hat, t_true):
    """

    Parameters
    ----------
    t_hat :
        
    t_true :
        

    Returns
    -------

    """
    return MSELoss()(t_hat, t_true)
