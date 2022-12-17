import logging

import numpy.random as rng
import torch

from torch import tensor
from madminer.utils.ml.models.base import BaseFlow

logger = logging.getLogger(__name__)


class BatchNorm(BaseFlow):
    """BatchNorm implementation"""

    def __init__(self, n_inputs, alpha=0.1, eps=1.0e-5):
        super().__init__(n_inputs)

        self.n_inputs = n_inputs
        self.alpha = alpha
        self.eps = eps

        # Running averages: will be created at first call of forward
        self.calculated_running_mean = False
        self.running_mean = torch.zeros(self.n_inputs)
        self.running_var = torch.zeros(self.n_inputs)

    def forward(self, x, fixed_params=False):
        """Calculates x -> u(x) (batch norming)"""

        # batch statistics
        if fixed_params:
            mean = self.running_mean
            var = self.running_var
        else:
            mean = torch.mean(x, dim=0)
            var = torch.mean((x - mean) ** 2, dim=0) + self.eps

            # keep track of running mean and var (for u -> x direction)
            if not self.calculated_running_mean:
                self.running_mean = mean
                self.running_var = var
            else:
                self.running_mean = (1.0 - self.alpha) * self.running_mean + self.alpha * mean
                self.running_var = (1.0 - self.alpha) * self.running_var + self.alpha * var
                self.calculated_running_mean = True

        # transformation
        u = (x - mean) / torch.sqrt(var)

        # log det du / dx
        logdet = -0.5 * torch.sum(torch.log(var))

        return u, logdet

    def inverse(self, u):
        """Calculates u -> x(u) (the approximate inverse transformation based on running mean and variance)"""

        return torch.sqrt(self.running_var) * u + self.running_mean

    def generate_samples(self, n_samples=1, u=None, **kwargs):
        if u is None:
            u = tensor(rng.randn(n_samples, self.n_inputs))

        return torch.sqrt(self.running_var) * u + self.running_mean

    def to(self, *args, **kwargs):
        logger.debug("Transforming BatchNorm to %s", args)

        self = super().to(*args, **kwargs)

        self.running_mean = self.running_mean.to(*args, **kwargs)
        self.running_var = self.running_var.to(*args, **kwargs)

        return self
