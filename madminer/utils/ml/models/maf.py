from __future__ import absolute_import, division, print_function

import numpy.random as rng
import logging

import torch.nn as nn
from torch import tensor

from madminer.utils.ml.models.base import BaseFlow, BaseConditionalFlow
from madminer.utils.ml.models.made import GaussianMADE, ConditionalGaussianMADE
from madminer.utils.ml.models.batch_norm import BatchNorm


class MaskedAutoregressiveFlow(BaseFlow):
    """ """
    def __init__(self, n_inputs, n_hiddens, n_mades, activation='relu', batch_norm=True,
                 input_order='sequential', mode='sequential', alpha=0.1):

        super(MaskedAutoregressiveFlow, self).__init__(n_inputs)

        # save input arguments
        self.n_inputs = n_inputs
        self.n_hiddens = n_hiddens
        self.n_mades = n_mades
        self.activation = activation
        self.batch_norm = batch_norm
        self.mode = mode
        self.alpha = alpha

        # Dtype and GPU / CPU management
        self.to_args = None
        self.to_kwargs = None

        # Build MADEs
        self.mades = nn.ModuleList()
        for i in range(n_mades):
            made = GaussianMADE(n_inputs, n_hiddens, activation=activation, input_order=input_order, mode=mode)
            self.mades.append(made)
            if not (isinstance(input_order, str) and input_order == 'random'):
                input_order = made.input_order[::-1]

        # Batch normalization
        self.bns = None
        if self.batch_norm:
            self.bns = nn.ModuleList()
            for i in range(n_mades):
                bn = BatchNorm(n_inputs, alpha=self.alpha)
                self.bns.append(bn)

    def forward(self, x, fix_batch_norm=False):
        # Change batch norm means only while training
        if not self.training:
            fix_batch_norm = True

        logdet_dudx = 0.0
        u = x

        for i, made in enumerate(self.mades):
            # inverse autoregressive transform
            u, this_logdet = made(u)

            logdet_dudx += this_logdet

            # batch normalization
            if self.batch_norm:
                bn = self.bns[i]
                u, this_logdet = bn(u, fixed_params=fix_batch_norm)
                logdet_dudx += this_logdet

        return u, logdet_dudx

    def generate_samples(self, n_samples=1, u=None, **kwargs):
        x = tensor(rng.randn(n_samples, self.n_inputs)) if u is None else u

        if self.to_args is not None or self.to_kwargs is not None:
            x = x.to(*self.to_args, **self.to_kwargs)

        if self.batch_norm:
            mades = [made for made in self.mades]
            bns = [bn for bn in self.bns]

            for i, (made, bn) in enumerate(zip(mades[::-1], bns[::-1])):
                x = bn.inverse(x)
                x = made.generate_samples(n_samples, x)
        else:
            mades = [made for made in self.mades]
            for made in mades[::-1]:
                x = made.generate_samples(n_samples, x)

        return x

    def to(self, *args, **kwargs):
        self.to_args = args
        self.to_kwargs = kwargs

        self = super().to(*args, **kwargs)

        for i, (made) in enumerate(self.mades):
            self.mades[i] = made.to(*args, **kwargs)

        if self.batch_norm:
            for i, (bn) in enumerate(self.bns):
                self.bns[i] = bn.to(*args, **kwargs)

        return self


class ConditionalMaskedAutoregressiveFlow(BaseConditionalFlow):
    """ """
    def __init__(self, n_conditionals, n_inputs, n_hiddens, n_mades, activation='relu', batch_norm=True,
                 input_order='sequential', mode='sequential', alpha=0.1):

        super(ConditionalMaskedAutoregressiveFlow, self).__init__(n_conditionals, n_inputs)

        # save input arguments
        self.n_conditionals = n_conditionals
        self.n_inputs = n_inputs
        self.n_hiddens = n_hiddens
        self.n_mades = n_mades
        self.activation = activation
        self.batch_norm = batch_norm
        self.mode = mode
        self.alpha = alpha

        # Dtype and GPU / CPU management
        self.to_args = None
        self.to_kwargs = None

        # Build MADEs
        self.mades = nn.ModuleList()
        for i in range(n_mades):
            made = ConditionalGaussianMADE(n_conditionals, n_inputs, n_hiddens, activation=activation,
                                           input_order=input_order, mode=mode)
            self.mades.append(made)
            if not (isinstance(input_order, str) and input_order != 'random'):
                input_order = made.input_order[::-1]

        # Batch normalizatino
        self.bns = None
        if self.batch_norm:
            self.bns = nn.ModuleList()
            for i in range(n_mades):
                bn = BatchNorm(n_inputs, alpha=self.alpha)
                self.bns.append(bn)

    def forward(self, theta, x, fix_batch_norm=None):
        """

        Parameters
        ----------
        theta :
            
        x :
            
        fix_batch_norm :
             (Default value = None)

        Returns
        -------

        """
        if x.shape[1] != self.n_inputs:
            logging.error('x has wrong shape: %s', x.shape)
            logging.debug('theta shape: %s', theta.shape)
            logging.debug('theta content: %s', theta)
            logging.debug('x content: %s', x)

            raise ValueError('Wrong x shape')

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

    def generate_samples(self, theta, u=None, **kwargs):
        """

        Parameters
        ----------
        theta :
            
        u :
             (Default value = None)
        **kwargs :
            

        Returns
        -------

        """
        n_samples = theta.shape[0]

        x = tensor(rng.randn(n_samples, self.n_inputs)) if u is None else u

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
        """

        Parameters
        ----------
        *args :
            
        **kwargs :
            

        Returns
        -------

        """
        self.to_args = args
        self.to_kwargs = kwargs

        self = super().to(*args, **kwargs)

        for i, (made) in enumerate(self.mades):
            self.mades[i] = made.to(*args, **kwargs)

        if self.batch_norm:
            for i, (bn) in enumerate(self.bns):
                self.bns[i] = bn.to(*args, **kwargs)

        return self
