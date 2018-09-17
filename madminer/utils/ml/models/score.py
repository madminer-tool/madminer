from __future__ import absolute_import, division, print_function, unicode_literals

import torch.nn as nn

from madminer.utils.ml.utils import get_activation_function


class LocalScoreEstimator(nn.Module):
    """ Module that implements local score estimators for methods like SALLY and SALLINO, or the calculation
     of Fisher information matrices. """

    def __init__(self, n_observables, n_parameters, n_hidden, activation='tanh'):

        super(LocalScoreEstimator, self).__init__()

        # Save input
        self.n_hidden = n_hidden
        self.activation = get_activation_function(activation)

        # Build network
        self.layers = nn.ModuleList()
        n_last = n_observables

        # Hidden layers
        for n_hidden_units in n_hidden:
            self.layers.append(
                nn.Linear(n_last, n_hidden_units)
            )
            n_last = n_hidden_units

        # Log r layer
        self.layers.append(
            nn.Linear(n_last, n_parameters)
        )

    def forward(self, x):
        t_hat = x

        for i, layer in enumerate(self.layers):
            if i > 0:
                t_hat = self.activation(t_hat)
            t_hat = layer(t_hat)

        return t_hat

    def to(self, *args, **kwargs):
        self = super(LocalScoreEstimator, self).to(*args, **kwargs)

        for i, layer in enumerate(self.layers):
            self.layers[i] = layer.to(*args, **kwargs)

        return self
