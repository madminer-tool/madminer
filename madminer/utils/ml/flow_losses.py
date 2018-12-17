from __future__ import absolute_import, division, print_function

import torch
from torch.nn.modules.loss import MSELoss
import logging

logger = logging.getLogger(__name__)


def negative_log_likelihood(log_p_pred, t_pred, t_true):
    return -torch.mean(log_p_pred)


def score_mse(log_p_pred, t_pred, t_true):
    return MSELoss()(t_pred, t_true)
