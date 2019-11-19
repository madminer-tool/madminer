from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn as nn
from torch.autograd import grad
from madminer.utils.ml.utils import get_activation_function
import logging

logger = logging.getLogger(__name__)


class DenseSingleParameterizedRatioModel(nn.Module):
    """ Module that implements agnostic parameterized likelihood estimators such as RASCAL or ALICES. Only the
    numerator of the ratio is parameterized. """

    def __init__(self, n_observables, n_parameters, n_hidden, activation="tanh", dropout_prob=0.0):

        super(DenseSingleParameterizedRatioModel, self).__init__()

        # Save input
        self.n_hidden = n_hidden
        self.activation = get_activation_function(activation)
        self.dropout_prob = dropout_prob

        # Build network
        self.layers = nn.ModuleList()
        n_last = n_observables + n_parameters

        # Hidden layers
        for n_hidden_units in n_hidden:
            if self.dropout_prob > 1.0e-9:
                self.layers.append(nn.Dropout(self.dropout_prob))
            self.layers.append(nn.Linear(n_last, n_hidden_units))
            n_last = n_hidden_units

        # Log r layer
        if self.dropout_prob > 1.0e-9:
            self.layers.append(nn.Dropout(self.dropout_prob))
        self.layers.append(nn.Linear(n_last, 1))

    def forward(self, theta, x, track_score=True, return_grad_x=False, create_gradient_graph=True):

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
        s_hat = 1.0 / (1.0 + torch.exp(log_r_hat))

        # Score t
        if track_score:
            t_hat, = grad(
                log_r_hat,
                theta,
                grad_outputs=torch.ones_like(log_r_hat.data),
                # grad_outputs=log_r_hat.data.new(log_r_hat.shape).fill_(1),
                only_inputs=True,
                create_graph=create_gradient_graph,
            )
        else:
            t_hat = None

        # Calculate gradient wrt x
        if return_grad_x:
            x_gradient, = grad(
                log_r_hat,
                x,
                grad_outputs=torch.ones_like(log_r_hat.data),
                only_inputs=True,
                create_graph=create_gradient_graph,
            )

            return s_hat, log_r_hat, t_hat, x_gradient

        return s_hat, log_r_hat, t_hat

    def to(self, *args, **kwargs):
        self = super(DenseSingleParameterizedRatioModel, self).to(*args, **kwargs)

        for i, layer in enumerate(self.layers):
            self.layers[i] = layer.to(*args, **kwargs)

        return self


class DenseDoublyParameterizedRatioModel(nn.Module):
    """ Module that implements agnostic parameterized likelihood estimators such as RASCAL or ALICES. Both
    numerator and denominator of the ratio are parameterized. """

    def __init__(self, n_observables, n_parameters, n_hidden, activation="tanh", dropout_prob=0.0):

        super(DenseDoublyParameterizedRatioModel, self).__init__()

        # Save input
        self.n_hidden = n_hidden
        self.activation = get_activation_function(activation)
        self.dropout_prob = dropout_prob

        # Build network
        self.layers = nn.ModuleList()
        n_last = n_observables + 2 * n_parameters

        # Hidden layers
        for n_hidden_units in n_hidden:
            if self.dropout_prob > 1.0e-9:
                self.layers.append(nn.Dropout(self.dropout_prob))
            self.layers.append(nn.Linear(n_last, n_hidden_units))
            n_last = n_hidden_units

        # Log r layer
        if self.dropout_prob > 1.0e-9:
            self.layers.append(nn.Dropout(self.dropout_prob))
        self.layers.append(nn.Linear(n_last, 1))

    def forward(self, theta0, theta1, x, track_score=True, return_grad_x=False, create_gradient_graph=True):

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
        s_hat = 1.0 / (1.0 + torch.exp(log_r_hat))

        # Score t
        if track_score:
            t_hat0, = grad(
                log_r_hat,
                theta0,
                grad_outputs=torch.ones_like(log_r_hat.data),
                only_inputs=True,
                create_graph=create_gradient_graph,
            )
            t_hat1, = grad(
                log_r_hat,
                theta1,
                grad_outputs=torch.ones_like(log_r_hat.data),
                only_inputs=True,
                create_graph=create_gradient_graph,
            )
            # NOTE: this is a factor of 4 slower than the simple parameterized version (2 gradients * 2 times
            #       slower calculation each)
        else:
            t_hat0 = None
            t_hat1 = None

        # Calculate gradient wrt x
        if return_grad_x:
            x_gradient, = grad(
                log_r_hat,
                x,
                grad_outputs=torch.ones_like(log_r_hat.data),
                only_inputs=True,
                create_graph=create_gradient_graph,
            )

            return s_hat, log_r_hat, t_hat0, t_hat1, x_gradient

        return s_hat, log_r_hat, t_hat0, t_hat1

    def to(self, *args, **kwargs):
        self = super(DenseDoublyParameterizedRatioModel, self).to(*args, **kwargs)

        for i, layer in enumerate(self.layers):
            self.layers[i] = layer.to(*args, **kwargs)

        return self


class DenseComponentRatioModel(nn.Module):
    """ Module that implements agnostic parameterized likelihood estimators such as RASCAL or ALICES. Only the
    numerator of the ratio is parameterized. """

    def __init__(self, n_observables, n_hidden, activation="tanh", dropout_prob=0.0):

        super(DenseComponentRatioModel, self).__init__()

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
        self.layers.append(nn.Linear(n_last, 1))

    def forward(self, x):

        # log r estimator
        log_r_hat = x

        for i, layer in enumerate(self.layers):
            if i > 0:
                log_r_hat = self.activation(log_r_hat)
            log_r_hat = layer(log_r_hat)

        return log_r_hat

    def to(self, *args, **kwargs):
        self = super(DenseComponentRatioModel, self).to(*args, **kwargs)

        for i, layer in enumerate(self.layers):
            self.layers[i] = layer.to(*args, **kwargs)

        return self


class DenseMorphingAwareRatioModel(nn.Module):
    def __init__(
        self, components, morphing_matrix, n_observables, n_parameters, n_hidden, activation="tanh", dropout_prob=0.0
    ):

        super(DenseMorphingAwareRatioModel, self).__init__()

        # Save input
        self.n_hidden = n_hidden
        self.activation = get_activation_function(activation)
        self.dropout_prob = dropout_prob
        self.n_components, self.n_parameters = components.shape

        # Morphing setup
        self.register_buffer("components", torch.tensor(components))
        self.register_buffer("morphing_matrix", torch.tensor(morphing_matrix))

        logger.debug("Loaded morphing matrix into PyTorch model:\n %s", morphing_matrix)

        # Build networks for all components
        self.component_estimators = nn.ModuleList()
        for _ in range(self.n_components):
            self.component_estimators.append(
                DenseComponentRatioModel(n_observables, n_hidden, activation, dropout_prob)
            )

    def forward(self, theta, x, track_score=True, return_grad_x=False, create_gradient_graph=True):

        """ Calculates estimated log likelihood ratio and the derived score. """

        # Track gradient wrt theta
        if track_score and not theta.requires_grad:
            theta.requires_grad = True

        # Track gradient wrt x
        if return_grad_x and not x.requires_grad:
            x.requires_grad = True

        # Calculate individual components
        log_r_hat_components = [component(x).unsqueeze(1) for component in self.component_estimators]
        log_r_hat_components = torch.cat(log_r_hat_components, 1)  # (batchsize, n_components, 1)

        # Calculate weights
        component_weights = []
        for c in range(self.n_components):
            component_weight = 1.0
            for p in range(self.n_parameters):
                component_weight = component_weight * (theta[:, p] ** self.components[c, p])
            component_weights.append(component_weight.unsqueeze(1))
        component_weights = torch.cat(component_weights, dim=1)  # (batchsize, n_components)

        # # Debugging
        # # morphing_matrix:  (n_benchmarks, n_components)
        # weights = torch.einsum("cn,bc->bn", [self.morphing_matrix, component_weights])  # (batchsize, n_benchmarks)
        # # logger.debug("Thetas -> weights:")
        # # for i in range(weights.size(0)):
        # #     logger.debug("  %s -> %s", theta[i].detach().numpy(), weights[i].detach().numpy())
        # logger.debug("Weights: %s", weights.detach().numpy())
        # logger.debug("Component predictions: %s", log_r_hat_components.detach().numpy())
        # logger.debug("Combined prediction: %s", log_r_hat.detach().numpy())

        # Put together
        log_r_hat = torch.einsum("cn,bc,bno->bo", [self.morphing_matrix, component_weights, log_r_hat_components])

        # Bayes-optimal s
        s_hat = 1.0 / (1.0 + torch.exp(log_r_hat))

        # Score t
        if track_score:
            t_hat, = grad(
                log_r_hat,
                theta,
                grad_outputs=torch.ones_like(log_r_hat.data),
                # grad_outputs=log_r_hat.data.new(log_r_hat.shape).fill_(1),
                only_inputs=True,
                create_graph=create_gradient_graph,
            )
        else:
            t_hat = None

        # Calculate gradient wrt x
        if return_grad_x:
            x_gradient, = grad(
                log_r_hat,
                x,
                grad_outputs=torch.ones_like(log_r_hat.data),
                only_inputs=True,
                create_graph=create_gradient_graph,
            )

            return s_hat, log_r_hat, t_hat, x_gradient

        return s_hat, log_r_hat, t_hat

    def to(self, *args, **kwargs):
        self = super(DenseMorphingAwareRatioModel, self).to(*args, **kwargs)

        self.components = self.components.to(*args, **kwargs)
        self.morphing_matrix = self.morphing_matrix.to(*args, **kwargs)
        for i, component in enumerate(self.component_estimators):
            self.component_estimators[i] = component.to(*args, **kwargs)

        return self
