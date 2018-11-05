from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn as nn
from torch.autograd import grad

from madminer.utils.ml.utils import get_activation_function


class ParameterizedRatioEstimator(nn.Module):
    """ Module that implements agnostic parameterized likelihood estimators such as RASCAL or ALICES. Only the
    numerator of the ratio is parameterized. """

    def __init__(self, n_observables, n_parameters, n_hidden, activation='tanh'):

        super(ParameterizedRatioEstimator, self).__init__()

        # Save input
        self.n_hidden = n_hidden
        self.activation = get_activation_function(activation)

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

    def forward(self, theta, x, track_score=True, return_grad_x=False):

        """ Calculates estimated log likelihood ratio and the derived score. """

        # Track gradient wrt theta
        if track_score and not theta.requires_grad:
            theta.requires_grad = True

        # Track gradient wrt x
        if return_grad_x and not x.requires_grad:
            x.requires_grad = True

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

        # Calculate gradient wrt x
        if return_grad_x:
            x_gradient = grad(log_r_hat, x,
                          grad_outputs=torch.ones_like(log_r_hat.data),
                          only_inputs=True, create_graph=True)[0]

            return s_hat, log_r_hat, t_hat, x_gradient

        return s_hat, log_r_hat, t_hat

    def to(self, *args, **kwargs):
        self = super(ParameterizedRatioEstimator, self).to(*args, **kwargs)

        for i, layer in enumerate(self.layers):
            self.layers[i] = layer.to(*args, **kwargs)

        return self


class DoublyParameterizedRatioEstimator(nn.Module):
    """ Module that implements agnostic parameterized likelihood estimators such as RASCAL or ALICES. Both
    numerator and denominator of the ratio are parameterized. """

    def __init__(self, n_observables, n_parameters, n_hidden, activation='tanh'):

        super(DoublyParameterizedRatioEstimator, self).__init__()

        # Save input
        self.n_hidden = n_hidden
        self.activation = get_activation_function(activation)

        # Build network
        self.layers = nn.ModuleList()
        n_last = n_observables + 2 * n_parameters

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

    def forward(self, theta0, theta1, x, track_score=True, return_grad_x=False):

        """ Calculates estimated log likelihood ratio and the derived score. """

        # Track gradient wrt thetas
        if track_score and not theta0.requires_grad:
            theta0.requires_grad = True
        if track_score and not theta1.requires_grad:
            theta1.requires_grad = True

        # Track gradient wrt x
        if return_grad_x and not x.requires_grad:
            x.requires_grad = True

        # f(x | theta0, theta1)
        f_th0_th1 = torch.cat((theta0, theta1, x), 1)

        for i, layer in enumerate(self.layers):
            if i > 0:
                f_th0_th1 = self.activation(f_th0_th1)
            f_th0_th1 = layer(f_th0_th1)

        # f(x | theta1, theta0)
        f_th1_th0 = torch.cat((theta1, theta0, x), 1)

        for i, layer in enumerate(self.layers):
            if i > 0:
                f_th1_th0 = self.activation(f_th1_th0)
            f_th1_th0 = layer(f_th1_th0)

        # Antisymmetric combination
        log_r_hat = f_th0_th1 - f_th1_th0

        # Bayes-optimal s
        s_hat = 1. / (1. + torch.exp(log_r_hat))

        # Score t
        if track_score:
            t_hat0 = grad(log_r_hat, theta0,
                          grad_outputs=torch.ones_like(log_r_hat.data),
                          only_inputs=True, create_graph=True)[0]
            t_hat1 = grad(log_r_hat, theta1,
                          grad_outputs=torch.ones_like(log_r_hat.data),
                          only_inputs=True, create_graph=True)[0]
            # NOTE: this is a factor of 4 slower than the simple parameterized version (2 gradients * 2 times
            #       slower calculation each)
        else:
            t_hat0 = None
            t_hat1 = None

        # Calculate gradient wrt x
        if return_grad_x:
            x_gradient = grad(log_r_hat, x,
                          grad_outputs=torch.ones_like(log_r_hat.data),
                          only_inputs=True, create_graph=True)[0]

            return s_hat, log_r_hat, t_hat0, t_hat1, x_gradient

        return s_hat, log_r_hat, t_hat0, t_hat1

    def to(self, *args, **kwargs):
        self = super(DoublyParameterizedRatioEstimator, self).to(*args, **kwargs)

        for i, layer in enumerate(self.layers):
            self.layers[i] = layer.to(*args, **kwargs)

        return self
