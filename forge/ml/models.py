from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn as nn
from torch.autograd import grad

from forge.ml.utils import get_activation


class ParameterizedRatioEstimator(nn.Module):
    """ Module that implements agnostic parameterized likelihood estimators such as RASCAL or ALICES """

    def __init__(self, n_observables, n_parameters, n_hidden, activation='tanh'):

        super(ParameterizedRatioEstimator, self).__init__()

        # Save input
        self.n_hidden = n_hidden
        self.activation = get_activation(activation)

        # Build network
        self.layers = nn.ModuleList()
        n_last = n_observables + n_parameters

        # Hidden layers
        for n_hidden_units in n_hidden:
            self.layers.append(
                nn.Linear(n_last, n_hidden_units)
            )
            n_last = n_hidden_units

        # Log r layer
        self.layers.append(
            nn.Linear(n_last, 1)
        )

    def forward(self, theta, x, track_score=True):

        """
        Calculates estimated log likelihood ratio and the derived score.

        :param theta:
        :param x:
        :param track_score:
        :return: s_hat, log_r_hat, t_hat
        """

        # Track gradient wrt theta
        if track_score and not theta.requires_grad:  # Can this happen?
            theta.requires_grad = True

        # log r estimator
        log_r_hat = torch.cat((theta, x), 1)

        for i, layer in enumerate(self.layers):
            if i > 0:
                log_r_hat = self.activation(log_r_hat)
            log_r_hat = layer(log_r_hat)

        # Bayes-optimal s
        s_hat = 1. / (1. + torch.exp(log_r_hat))

        # Score t
        if track_score:
            t_hat = grad(log_r_hat, theta,
                         grad_outputs=torch.ones_like(log_r_hat.data),
                         only_inputs=True, create_graph=True)[0]
        else:
            t_hat = None

        return s_hat, log_r_hat, t_hat
