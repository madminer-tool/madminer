import logging
import torch
import torch.nn as nn

from madminer.utils.ml.utils import get_activation_function
from torch.autograd import grad

logger = logging.getLogger(__name__)


class DenseLocalScoreModel(nn.Module):
    """Module that implements local score estimators for methods like SALLY and SALLINO, or the calculation
    of Fisher information matrices."""

    def __init__(self, n_observables, n_parameters, n_hidden, activation="tanh", dropout_prob=0.0):
        super().__init__()

        # Save input
        self.n_hidden = n_hidden
        self.activation = get_activation_function(activation)
        self.dropout_prob = dropout_prob

        # Build network
        self.layers = nn.ModuleList()
        n_last = n_observables

        # Hidden layers
        for n_hidden_units in n_hidden:
            if self.dropout_prob > 1.0e-9:
                self.layers.append(nn.Dropout(self.dropout_prob))
            self.layers.append(nn.Linear(n_last, n_hidden_units))
            n_last = n_hidden_units

        # Log r layer
        if self.dropout_prob > 1.0e-9:
            self.layers.append(nn.Dropout(self.dropout_prob))
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
            x_gradient = grad(
                t_hat,
                x,
                grad_outputs=torch.ones_like(t_hat.data),
                only_inputs=True,
                create_graph=True,
            )[0]

            return t_hat, x_gradient

        return t_hat

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)

        for i, layer in enumerate(self.layers):
            self.layers[i] = layer.to(*args, **kwargs)

        return self
