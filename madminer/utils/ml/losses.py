import logging

import numpy as np
import torch
from torch.nn import BCELoss
from torch.nn import MSELoss

logger = logging.getLogger(__name__)


def ratio_mse_num(s_hat, log_r_hat, t0_hat, t1_hat, y_true, r_true, t0_true, t1_true, log_r_clip=10.0):
    r_true = torch.clamp(r_true, np.exp(-log_r_clip), np.exp(log_r_clip))
    log_r_hat = torch.clamp(log_r_hat, -log_r_clip, log_r_clip)

    inverse_r_hat = torch.exp(-log_r_hat)
    return MSELoss()((1.0 - y_true) * inverse_r_hat, (1.0 - y_true) * (1.0 / r_true))


def ratio_mse_den(s_hat, log_r_hat, t0_hat, t1_hat, y_true, r_true, t0_true, t1_true, log_r_clip=10.0):
    r_true = torch.clamp(r_true, np.exp(-log_r_clip), np.exp(log_r_clip))
    log_r_hat = torch.clamp(log_r_hat, -log_r_clip, log_r_clip)

    r_hat = torch.exp(log_r_hat)
    return MSELoss()(y_true * r_hat, y_true * r_true)


def ratio_mse(s_hat, log_r_hat, t0_hat, t1_hat, y_true, r_true, t0_true, t1_true, log_r_clip=10.0):
    return ratio_mse_num(
        s_hat, log_r_hat, t0_hat, t1_hat, y_true, r_true, t0_true, t1_true, log_r_clip
    ) + ratio_mse_den(s_hat, log_r_hat, t0_hat, t1_hat, y_true, r_true, t0_true, t1_true, log_r_clip)


def ratio_score_mse_num(s_hat, log_r_hat, t0_hat, t1_hat, y_true, r_true, t0_true, t1_true):
    return MSELoss()((1.0 - y_true) * t0_hat, (1.0 - y_true) * t0_true)


def ratio_score_mse_den(s_hat, log_r_hat, t0_hat, t1_hat, y_true, r_true, t0_true, t1_true):
    return MSELoss()(y_true * t1_hat, y_true * t1_true)


def ratio_score_mse(s_hat, log_r_hat, t0_hat, t1_hat, y_true, r_true, t0_true, t1_true):
    return ratio_score_mse_num(
        s_hat, log_r_hat, t0_hat, t1_hat, y_true, r_true, t0_true, t1_true
    ) + ratio_score_mse_den(s_hat, log_r_hat, t0_hat, t1_hat, y_true, r_true, t0_true, t1_true)


def ratio_xe(s_hat, log_r_hat, t0_hat, t1_hat, y_true, r_true, t0_true, t1_true):
    s_hat = 1.0 / (1.0 + torch.exp(log_r_hat))

    return BCELoss()(s_hat, y_true)


def ratio_augmented_xe(s_hat, log_r_hat, t0_hat, t1_hat, y_true, r_true, t0_true, t1_true):
    s_hat = 1.0 / (1.0 + torch.exp(log_r_hat))
    s_true = 1.0 / (1.0 + r_true)

    return BCELoss()(s_hat, s_true)


def local_score_mse(t_hat, t_true):
    return MSELoss()(t_hat, t_true)


def flow_nll(log_p_pred, t_pred, t_true):
    return -torch.mean(log_p_pred)


def flow_score_mse(log_p_pred, t_pred, t_true):
    return MSELoss()(t_pred, t_true)
