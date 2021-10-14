import numpy as np
import torch
import torch.nn as nn

from abc import abstractmethod
from torch.autograd import grad


class BaseFlow(nn.Module):
    """ """

    def __init__(self, n_inputs, **kwargs):
        super().__init__()
        self.n_inputs = n_inputs

    @abstractmethod
    def forward(self, x, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def generate_samples(self, n_samples=1, u=None, **kwargs):
        raise NotImplementedError()

    def log_likelihood(self, x, **kwargs):
        """ Calculates log p(x) with a Gaussian base density """

        u, logdet_dudx = self.forward(x, **kwargs)

        constant = float(-0.5 * self.n_inputs * np.log(2.0 * np.pi))
        log_likelihood = constant - 0.5 * torch.sum(u ** 2, dim=1) + logdet_dudx

        return u, log_likelihood

    def log_likelihood_and_score(self, x, **kwargs):
        """ Calculates log p(x) and t(x) with a Gaussian base density """

        u, log_likelihood = self.log_likelihood(x, **kwargs)

        return u, log_likelihood, None


class BaseConditionalFlow(nn.Module):

    def __init__(self, n_conditionals, n_inputs, **kwargs):
        super().__init__()
        self.n_conditionals = n_conditionals
        self.n_inputs = n_inputs

    @abstractmethod
    def forward(self, theta, x, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def generate_samples(self, theta, u=None, **kwargs):
        raise NotImplementedError()

    def log_likelihood(self, theta, x, **kwargs):
        """ Calculates u(x) and log p(x) with a Gaussian base density """

        u, logdet_dudx = self.forward(theta, x, **kwargs)

        constant = float(-0.5 * self.n_inputs * np.log(2.0 * np.pi))
        log_likelihood = constant - 0.5 * torch.sum(u ** 2, dim=1) + logdet_dudx

        return u, log_likelihood

    def log_likelihood_and_score(self, theta, x, **kwargs):
        """ Calculates u(x), log p(x), and the score t(x) with a Gaussian base density """

        if theta.shape[0] == 1:
            theta = theta.expand(x.shape[0], -1)

        if not theta.requires_grad:
            theta.requires_grad = True

        u, log_likelihood = self.log_likelihood(theta, x, **kwargs)

        score = grad(
            log_likelihood,
            theta,
            grad_outputs=torch.ones_like(log_likelihood.data),
            only_inputs=True,
            create_graph=True,
        )[0]

        return u, log_likelihood, score
