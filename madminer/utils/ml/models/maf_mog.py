import logging
import torch.nn as nn

from madminer.utils.ml.models.base import BaseConditionalFlow
from madminer.utils.ml.models.batch_norm import BatchNorm
from madminer.utils.ml.models.made import ConditionalGaussianMADE
from madminer.utils.ml.models.made_mog import ConditionalMixtureMADE

logger = logging.getLogger(__name__)


class ConditionalMixtureMaskedAutoregressiveFlow(BaseConditionalFlow):
    """ """

    def __init__(
        self,
        n_conditionals,
        n_inputs,
        n_hiddens,
        n_mades,
        n_components=10,
        activation="relu",
        batch_norm=True,
        input_order="sequential",
        mode="sequential",
        alpha=0.1,
    ):

        super().__init__(n_conditionals, n_inputs)

        # save input arguments
        self.n_conditionals = n_conditionals
        self.n_inputs = n_inputs
        self.n_hiddens = n_hiddens
        self.n_mades = n_mades
        self.activation = activation
        self.batch_norm = batch_norm
        self.mode = mode
        self.alpha = alpha
        self.n_components = n_components

        # Dtype and GPU / CPU management
        self.to_args = None
        self.to_kwargs = None

        # Build MADEs
        self.mades = nn.ModuleList()
        for i in range(n_mades - 1):
            made = ConditionalGaussianMADE(
                n_conditionals, n_inputs, n_hiddens, activation=activation, input_order=input_order, mode=mode
            )
            self.mades.append(made)
            if not (isinstance(input_order, str) and input_order != "random"):
                input_order = made.input_order[::-1]

        # Last MADE MoG
        self.made_mog = ConditionalMixtureMADE(
            n_conditionals,
            n_inputs,
            n_hiddens,
            n_components=n_components,
            activation=activation,
            input_order=input_order,
            mode=mode,
        )

        # Batch normalization
        self.bns = None
        if self.batch_norm:
            self.bns = nn.ModuleList()
            for i in range(n_mades):
                bn = BatchNorm(n_inputs, alpha=self.alpha)
                self.bns.append(bn)

    def forward(self, theta, x, fix_batch_norm=None):
        if x.shape[1] != self.n_inputs:
            logger.error("x has wrong shape: %s", x.shape)
            logger.debug("theta shape: %s", theta.shape)
            logger.debug("theta content: %s", theta)
            logger.debug("x content: %s", x)

            raise ValueError("Wrong x shape")

        # Change batch norm means only while training
        if fix_batch_norm is None:
            fix_batch_norm = not self.training

        logdet_dudx = 0.0
        u = x

        for i, made in enumerate(self.mades):
            # inverse autoregressive transform
            u, this_logdet = made(theta, u)
            logdet_dudx += this_logdet

            # batch normalization
            if self.batch_norm:
                bn = self.bns[i]
                u, this_logdet = bn(u, fixed_params=fix_batch_norm)
                logdet_dudx += this_logdet

        return u, logdet_dudx

    def log_likelihood(self, theta, x, **kwargs):
        """Calculates u(x) and log p(x) with a MADE MoG base density"""

        # MADEs and BNs
        u, logdet_dudx = self.forward(theta, x, **kwargs)

        # MADE MoG base density
        _, base_log_likelihood = self.made_mog.log_likelihood(theta, u)

        # Combine
        log_likelihood = base_log_likelihood + logdet_dudx

        return u, log_likelihood

    def generate_samples(self, theta, u=None, **kwargs):
        x = self.made_mog.generate_samples(theta, u, **kwargs)

        if self.batch_norm:
            mades = [made for made in self.mades]
            bns = [bn for bn in self.bns]

            for i, (made, bn) in enumerate(zip(mades[::-1], bns[::-1])):
                x = bn.inverse(x)
                x = made.generate_samples(theta, x)
        else:
            mades = [made for made in self.mades]
            for made in mades[::-1]:
                x = made.generate_samples(theta, x)

        return x

    def to(self, *args, **kwargs):
        self.to_args = args
        self.to_kwargs = kwargs

        self = super().to(*args, **kwargs)

        for i, (made) in enumerate(self.mades):
            self.mades[i] = made.to(*args, **kwargs)

        self.made_mog = self.made_mog.to(*args, **kwargs)

        if self.batch_norm:
            for i, (bn) in enumerate(self.bns):
                self.bns[i] = bn.to(*args, **kwargs)

        return self
