import torch
from torch.nn.modules.loss import MSELoss


def negative_log_likelihood(model, t_true):
    return -torch.mean(model.log_likelihood)


def score_mse(model, t_true):
    return MSELoss()(model.score, t_true)
