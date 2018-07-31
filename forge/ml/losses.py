import torch
from torch.nn.modules.loss import MSELoss


def ratio_mse(s_hat, log_r_hat, t_hat, y_true, r_true, t_true):
    r_hat = torch.exp(log_r_hat)
    return MSELoss()(r_hat, r_true)


def augmented_cross_entropy(s_hat, log_r_hat, t_hat, y_true, r_true, t_true):
    raise NotImplementedError


def score_mse(s_hat, log_r_hat, t_hat, y_true, r_true, t_true):
    return MSELoss()(t_hat, t_true)

