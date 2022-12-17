import logging

import numpy as np
import torch
import torch.nn as nn

from torch.autograd import grad
from madminer.utils.ml.utils import get_activation_function

logger = logging.getLogger(__name__)


class DenseSingleParameterizedRatioModel(nn.Module):
    """Module that implements agnostic parameterized likelihood estimators such as RASCAL or ALICES. Only the
    numerator of the ratio is parameterized."""

    def __init__(self, n_observables, n_parameters, n_hidden, activation="tanh", dropout_prob=0.0):
        super().__init__()

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
        self.append = self.layers.append(nn.Linear(n_last, 1))

    def forward(self, theta, x, track_score=True, return_grad_x=False, create_gradient_graph=True):
        """Calculates estimated log likelihood ratio and the derived score."""

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
            (t_hat,) = grad(
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
            (x_gradient,) = grad(
                log_r_hat,
                x,
                grad_outputs=torch.ones_like(log_r_hat.data),
                only_inputs=True,
                create_graph=create_gradient_graph,
            )

            return s_hat, log_r_hat, t_hat, x_gradient

        return s_hat, log_r_hat, t_hat

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)

        for i, layer in enumerate(self.layers):
            self.layers[i] = layer.to(*args, **kwargs)

        return self


class DenseDoublyParameterizedRatioModel(nn.Module):
    """Module that implements agnostic parameterized likelihood estimators such as RASCAL or ALICES. Both
    numerator and denominator of the ratio are parameterized."""

    def __init__(self, n_observables, n_parameters, n_hidden, activation="tanh", dropout_prob=0.0):
        super().__init__()

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
        """Calculates estimated log likelihood ratio and the derived score."""

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
            (t_hat0,) = grad(
                log_r_hat,
                theta0,
                grad_outputs=torch.ones_like(log_r_hat.data),
                only_inputs=True,
                create_graph=create_gradient_graph,
            )
            (t_hat1,) = grad(
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
            (x_gradient,) = grad(
                log_r_hat,
                x,
                grad_outputs=torch.ones_like(log_r_hat.data),
                only_inputs=True,
                create_graph=create_gradient_graph,
            )

            return s_hat, log_r_hat, t_hat0, t_hat1, x_gradient

        return s_hat, log_r_hat, t_hat0, t_hat1

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)

        for i, layer in enumerate(self.layers):
            self.layers[i] = layer.to(*args, **kwargs)

        return self


class DenseComponentRatioModel(nn.Module):
    """Module that implements agnostic parameterized likelihood estimators such as RASCAL or ALICES. Only the
    numerator of the ratio is parameterized."""

    def __init__(self, n_observables, n_hidden, activation="tanh", dropout_prob=0.0):
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
        self = super().to(*args, **kwargs)

        for i, layer in enumerate(self.layers):
            self.layers[i] = layer.to(*args, **kwargs)

        return self


class DenseMorphingAwareRatioModel(nn.Module):
    def __init__(
        self,
        components,
        morphing_matrix,
        n_observables,
        n_parameters,
        n_hidden,
        activation="tanh",
        dropout_prob=0.0,
        clamp_component_ratios=5.0,
    ):

        super().__init__()

        # Save input
        self.n_hidden = n_hidden
        self.activation = get_activation_function(activation)
        self.dropout_prob = dropout_prob
        self.n_components, self.n_parameters = components.shape
        self.clamp_component_ratios = clamp_component_ratios

        # Morphing setup
        self.register_buffer("components", torch.tensor(components))
        self.register_buffer("morphing_matrix", torch.tensor(morphing_matrix))

        logger.debug("Loaded morphing matrix into PyTorch model:\n %s", morphing_matrix)

        # Build networks for all components
        self.dsigma_component_estimators = nn.ModuleList()
        for _ in range(self.n_components):
            self.dsigma_component_estimators.append(
                DenseComponentRatioModel(n_observables, n_hidden, activation, dropout_prob)
            )

        self.log_sigma_ratio_components = torch.nn.Parameter(torch.zeros((self.n_components, 1)))  # (n_components, 1)

    def forward(self, theta, x, track_score=True, return_grad_x=False, create_gradient_graph=True):
        """Calculates estimated log likelihood ratio and the derived score."""

        # Track gradient wrt theta
        if track_score and not theta.requires_grad:
            theta.requires_grad = True

        # Track gradient wrt x
        if return_grad_x and not x.requires_grad:
            x.requires_grad = True

        # Calculate individual components dsigma_c(x) / dsigma1
        dsigma_ratio_components = [component(x).unsqueeze(1) for component in self.dsigma_component_estimators]
        dsigma_ratio_components = torch.cat(dsigma_ratio_components, 1)  # (batchsize, n_components, 1)
        dsigma_ratio_components = torch.exp(
            torch.clamp(dsigma_ratio_components, -self.clamp_component_ratios, self.clamp_component_ratios)
        )

        # Denominator (for changes in total xsecs with theta)
        sigma_ratio_components = torch.exp(
            torch.clamp(self.log_sigma_ratio_components, -self.clamp_component_ratios, self.clamp_component_ratios)
        )

        # Calculate morphing weights
        component_weights = []
        for c in range(self.n_components):
            component_weight = 1.0
            for p in range(self.n_parameters):
                component_weight = component_weight * (theta[:, p] ** self.components[c, p])
            component_weights.append(component_weight.unsqueeze(1))
        component_weights = torch.cat(component_weights, dim=1)  # (batchsize, n_components)

        # Put together
        weights = torch.einsum("cn,bc->bn", [self.morphing_matrix, component_weights])  # (batchsize, n_benchmarks)
        numerator = torch.einsum("bn,bno->bo", [weights, dsigma_ratio_components])
        denominator = torch.einsum("bn,no->bo", [weights, sigma_ratio_components])

        numerator = torch.clamp(numerator, np.exp(-self.clamp_component_ratios), np.exp(self.clamp_component_ratios))
        denominator = torch.clamp(
            denominator, np.exp(-self.clamp_component_ratios), np.exp(self.clamp_component_ratios)
        )
        r_hat = numerator / denominator

        # # Debugging
        # try:
        #     check_for_nonpos("Morphing-aware model: numerator", numerator)
        #     check_for_nonpos("Morphing-aware model: denominator", denominator)
        #     check_for_nonpos("Morphing-aware model: ratio", r_hat)
        # except NanException:
        #     logger.error("Inconsistent inputs in the morphing-aware model.")
        #
        #     filter = torch.where((r_hat <= 0.).any(axis = 1))
        #
        #     logger.error(  "x = %s", x[filter])
        #     logger.error(  "theta = %s", theta[filter])
        #     logger.error(  "morphing_matrix = %s", self.morphing_matrix)
        #     logger.error(  "dsigma_ratio_components = %s", dsigma_ratio_components[filter])
        #     logger.error(  "sigma_ratio_components = %s", sigma_ratio_components)
        #     logger.error(  "component_weights = %s", component_weights[filter])
        #     logger.error(  "numerator = %s", numerator[filter])
        #     logger.error(  "denominator = %s", denominator[filter])
        #     logger.error(  "ratio = %s", r_hat[filter])
        #
        #     logger.error("Let's go into this calculation a bit more:")
        #     # morphing_matrix:  (n_benchmarks, n_components)
        #     weights = torch.einsum("cn,bc->bn", [self.morphing_matrix, component_weights])  # (batchsize, n_benchmarks)
        #     logger.error("Thetas -> weights:")
        #     for i in range(weights.size(0)):
        #         logger.error("  %s -> %s", theta[i].detach().numpy(), weights[i].detach().numpy())
        #     logger.error("Weights: %s", weights.detach().numpy())
        #     logger.error("Component predictions for numerator: %s", sigma_ratio_components.detach().numpy())
        #     logger.error("Combined numerator: %s", numerator.detach().numpy())
        #
        #     raise

        # Bayes-optimal s
        r_hat = torch.clamp(r_hat, np.exp(-self.clamp_component_ratios), np.exp(self.clamp_component_ratios))
        log_r_hat = torch.log(torch.clamp(r_hat, 1.0e-6, 1.0e6))
        s_hat = 1.0 / (1.0 + r_hat)

        # Score t
        if track_score:
            (t_hat,) = grad(
                log_r_hat,
                theta,
                grad_outputs=torch.ones_like(log_r_hat.data),
                only_inputs=True,
                create_graph=create_gradient_graph,
            )
        else:
            t_hat = None

        # Calculate gradient wrt x
        if return_grad_x:
            (x_gradient,) = grad(
                log_r_hat,
                x,
                grad_outputs=torch.ones_like(log_r_hat.data),
                only_inputs=True,
                create_graph=create_gradient_graph,
            )

            return s_hat, log_r_hat, t_hat, x_gradient

        return s_hat, log_r_hat, t_hat


class DenseQuadraticMorphingAwareRatioModel(nn.Module):
    def __init__(self, n_observables, n_parameters, n_hidden, activation="tanh", dropout_prob=0.0):
        super().__init__()

        assert n_parameters == 1

        # Save input
        self.n_hidden = n_hidden
        self.activation = get_activation_function(activation)
        self.dropout_prob = dropout_prob

        # Build networks for all components
        self.dsigma_a = DenseComponentRatioModel(n_observables, n_hidden, activation, dropout_prob)
        self.dsigma_b = DenseComponentRatioModel(n_observables, n_hidden, activation, dropout_prob)
        self.sigma_a = torch.nn.Parameter(torch.ones((1,)))
        self.sigma_b = torch.nn.Parameter(torch.ones((1,)))

    def forward(self, theta, x, track_score=True, return_grad_x=False, create_gradient_graph=True):
        """Calculates estimated log likelihood ratio and the derived score."""

        assert theta.size()[1] == 1

        # Track gradient wrt theta
        if track_score and not theta.requires_grad:
            theta.requires_grad = True

        # Track gradient wrt x
        if return_grad_x and not x.requires_grad:
            x.requires_grad = True

        # Calculate individual components
        dsigma_a = self.dsigma_a(x)  # (batch, 1)
        dsigma_b = self.dsigma_b(x)  # (batch, 1)

        # Put together
        dsigma_ratio = (1 + theta * dsigma_a) ** 2 + (theta * dsigma_b) ** 2
        sigma_ratio = (1 + theta * self.sigma_a) ** 2 + (theta * self.sigma_b) ** 2
        r_hat = dsigma_ratio / sigma_ratio

        # Bayes-optimal s
        log_r_hat = torch.log(r_hat + 1.0e-9)
        s_hat = 1.0 / (1.0 + r_hat)

        # Score t
        if track_score:
            (t_hat,) = grad(
                log_r_hat,
                theta,
                grad_outputs=torch.ones_like(log_r_hat.data),
                only_inputs=True,
                create_graph=create_gradient_graph,
            )
        else:
            t_hat = None

        # Calculate gradient wrt x
        if return_grad_x:
            (x_gradient,) = grad(
                log_r_hat,
                x,
                grad_outputs=torch.ones_like(log_r_hat.data),
                only_inputs=True,
                create_graph=create_gradient_graph,
            )

            return s_hat, log_r_hat, t_hat, x_gradient

        return s_hat, log_r_hat, t_hat
