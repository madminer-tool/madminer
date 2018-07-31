import torch
from torch.nn import BCELoss, MSELoss


def ratio_mse_num(s_hat, log_r_hat, t_hat, y_true, r_true, t_true):
    inverse_r_hat = torch.exp(- log_r_hat)
    return MSELoss()((1. - y_true) * inverse_r_hat, (1. - y_true) * (1. / r_true))


def ratio_mse_den(s_hat, log_r_hat, t_hat, y_true, r_true, t_true):
    r_hat = torch.exp(log_r_hat)
    return MSELoss()(y_true * r_hat, y_true * r_true)


def ratio_mse(s_hat, log_r_hat, t_hat, y_true, r_true, t_true):
    return (ratio_mse_num(s_hat, log_r_hat, t_hat, y_true, r_true, t_true)
            + ratio_mse_den(s_hat, log_r_hat, t_hat, y_true, r_true, t_true))


def score_mse_num(s_hat, log_r_hat, t_hat, y_true, r_true, t_true):
    return MSELoss()((1. - y_true) * t_hat, (1. - y_true) * t_true)


def standard_cross_entropy(s_hat, log_r_hat, t_hat, y_true, r_true, t_true):
    s_hat = 1. / (1. + torch.exp(log_r_hat))

    return BCELoss(s_hat, y_true)


def augmented_cross_entropy(s_hat, log_r_hat, t_hat, y_true, r_true, t_true):
    s_hat = 1. / (1. + torch.exp(log_r_hat))
    s_true = 1. / (1. + r_true)

    return BCELoss(s_hat, s_true)
