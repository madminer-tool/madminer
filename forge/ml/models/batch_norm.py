import numpy as np

import torch
import torch.nn as nn


class BatchNorm(nn.Module):
    """ BatchNorm implementation """

    def __init__(self, n_units, alpha=0.1, eps=1.e-5):

        super(BatchNorm, self).__init__()

        self.n_units = n_units
        self.alpha = alpha
        self.eps = eps

        # Batch averages
        self.mean = None
        self.var = None

        # Running averages: will be created at first call of forward
        self.running_mean = None
        self.running_var = None

    def forward(self, x, fixed_params=False):

        """ Calculates x -> u(x) (batch norming) """

        # batch statistics
        if fixed_params:
            self.mean = self.running_mean
            self.var = self.running_var
        else:
            self.mean = torch.mean(x, dim=0)
            self.var = torch.mean((x - self.mean) ** 2, dim=0) + self.eps

            # keep track of running mean and var (for u -> x direction)
            if self.running_mean is None:
                self.running_mean = torch.zeros(self.n_units)
                self.running_var = torch.zeros(self.n_units)
                self.running_mean += self.mean
                self.running_var += self.var
            else:
                self.running_mean = (1. - self.alpha) * self.running_mean + self.alpha * self.mean
                self.running_var = (1. - self.alpha) * self.running_var + self.alpha * self.var

        # transformation
        u = (x - self.mean) / torch.sqrt(self.var)

        return u

    def inverse(self, u):

        """ Calculates u -> x(u) (the approximate inverse transformation based on running mean and variance) """

        x = torch.sqrt(self.running_var) * u + self.running_mean

        return x
