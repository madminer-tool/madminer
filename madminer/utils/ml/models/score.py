from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn as nn
from torch.autograd import grad
from madminer.utils.ml.utils import get_activation_function
import logging

logger = logging.getLogger(__name__)


class DenseLocalScoreModel(nn.Module):
    """Module that implements local score estimators for methods like SALLY and SALLINO, or the calculation
     of Fisher information matrices."""

    def __init__(self, n_observables, n_parameters, n_hidden, activation="tanh"):

        super(DenseLocalScoreModel, self).__init__()

        # Save input
        self.n_hidden = n_hidden
        self.activation = get_activation_function(activation)

        # Build network
        self.layers = nn.ModuleList()
        n_last = n_observables

        # Hidden layers
        for n_hidden_units in n_hidden:
            self.layers.append(nn.Linear(n_last, n_hidden_units))
            n_last = n_hidden_units

        # Log r layer
        self.layers.append(nn.Linear(n_last, n_parameters))

    def forward(self, x, return_grad_x=False):
        # Track gradient wrt x
        if return_grad_x and not x.requires_grad:
            x.requires_grad = True

        # Forward pass
        t_hat = x

        for i, layer in enumerate(self.layers):
            if i > 0:
                t_hat = self.activation(t_hat)
            t_hat = layer(t_hat)

        # Calculate gradient
        if return_grad_x:
            x_gradient = grad(t_hat, x, grad_outputs=torch.ones_like(t_hat.data), only_inputs=True, create_graph=True)[
                0
            ]

            return t_hat, x_gradient

        return t_hat

    def to(self, *args, **kwargs):
        self = super(DenseLocalScoreModel, self).to(*args, **kwargs)

        for i, layer in enumerate(self.layers):
            self.layers[i] = layer.to(*args, **kwargs)

        return self
