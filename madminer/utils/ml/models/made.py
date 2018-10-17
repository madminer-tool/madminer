from __future__ import absolute_import, division, print_function

import numpy as np
import numpy.random as rng
import logging

import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F

from madminer.utils.ml.models.base import BaseFlow, BaseConditionalFlow
from madminer.utils.ml.models.masks import create_degrees, create_masks, create_weights, create_weights_conditional
from madminer.utils.ml.utils import get_activation_function


class GaussianMADE(BaseFlow):
    """ """

    def __init__(self, n_inputs, n_hiddens, activation='relu', input_order='sequential', mode='sequential'):
        super(GaussianMADE, self).__init__(n_inputs)

        # save input arguments
        self.activation = activation
        self.n_inputs = n_inputs
        self.n_hiddens = n_hiddens
        self.mode = mode

        # create network's parameters
        self.degrees = create_degrees(n_inputs, n_hiddens, input_order, mode)
        self.Ms, self.Mmp = create_masks(self.degrees)
        self.Ws, self.bs, self.Wm, self.bm, self.Wp, self.bp = create_weights(n_inputs, n_hiddens, None)
        self.input_order = self.degrees[0]

        self.activation_function = get_activation_function(activation)

        # Output info
        self.m = None
        self.logp = None

        # Dtype and GPU / CPU management
        self.to_args = None
        self.to_kwargs = None

    def forward(self, x, **kwargs):
        """

        Parameters
        ----------
        x :
            
        **kwargs :
            

        Returns
        -------

        """
        # Conditioner
        h = x

        for M, W, b in zip(self.Ms, self.Ws, self.bs):
            try:
                h = self.activation_function(F.linear(h, torch.t(M * W), b))
            except (RuntimeError, AttributeError):
                logging.error('Abort! Abort!')
                logging.info('MADE settings: n_inputs = %s', self.n_inputs)
                logging.info('Shapes: x %s, h %s, M %s, W %s, b %s',
                             x.shape, h.shape, M.shape, W.shape, b.shape)
                logging.info('Types: x %s, h %s, M %s, W %s, b %s',
                             type(x), type(h), type(M), type(W), type(b))
                logging.info('CUDA: x %s, h %s, M %s, W %s, b %s',
                             x.is_cuda, h.is_cuda, M.is_cuda, W.is_cuda, b.is_cuda)
                raise

        # Gaussian parameters
        self.m = F.linear(h, torch.t(self.Mmp * self.Wm), self.bm)
        self.logp = F.linear(h, torch.t(self.Mmp * self.Wp), self.bp)

        # u(x)
        u = torch.exp(0.5 * self.logp) * (x - self.m)

        # log det du / dx
        logdet_dudx = 0.5 * torch.sum(self.logp, dim=1)

        return u, logdet_dudx

    def generate_samples(self, n_samples=1, u=None, **kwargs):
        """

        Parameters
        ----------
        n_samples :
             (Default value = 1)
        u :
             (Default value = None)
        **kwargs :
            

        Returns
        -------

        """
        x = torch.zeros([n_samples, self.n_inputs])
        if u is None:
            u = tensor(rng.randn(n_samples, self.n_inputs))

        if self.to_args is not None or self.to_kwargs is not None:
            x = x.to(*self.to_args, **self.to_kwargs)
            u = u.to(*self.to_args, **self.to_kwargs)

        for i in range(1, self.n_inputs + 1):
            self.forward(x)  # Sets Gaussian parameters: self.m and self.logp

            idx = np.argwhere(self.input_order == i)[0, 0]

            mask = torch.zeros([n_samples, self.n_inputs])
            if self.to_args is not None or self.to_kwargs is not None:
                mask = mask.to(*self.to_args, **self.to_kwargs)

            mask[:, idx] = 1.

            x = (1. - mask) * x + mask * (self.m + torch.exp(torch.clamp(-0.5 * self.logp, -10., 10.)) * u)

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

        for i, (M, W, b) in enumerate(zip(self.Ms, self.Ws, self.bs)):
            self.Ms[i] = M.to(*args, **kwargs)
            self.Ws[i] = nn.Parameter(W.to(*args, **kwargs))
            self.bs[i] = nn.Parameter(b.to(*args, **kwargs))

        self.Mmp = self.Mmp.to(*args, **kwargs)
        self.Wm = nn.Parameter(self.Wm.to(*args, **kwargs))
        self.bm = nn.Parameter(self.bm.to(*args, **kwargs))
        self.Wp = nn.Parameter(self.Wp.to(*args, **kwargs))
        self.bp = nn.Parameter(self.bp.to(*args, **kwargs))

        return self


class ConditionalGaussianMADE(BaseConditionalFlow):
    """ """
    def __init__(self, n_conditionals, n_inputs, n_hiddens, activation='relu', input_order='sequential',
                 mode='sequential'):
        super(ConditionalGaussianMADE, self).__init__(n_conditionals, n_inputs)

        # save input arguments
        self.activation = activation
        self.n_conditionals = n_conditionals
        self.n_inputs = n_inputs
        self.n_hiddens = n_hiddens
        self.mode = mode

        # create network's parameters
        self.degrees = create_degrees(n_inputs, n_hiddens, input_order, mode)
        self.Ms, self.Mmp = create_masks(self.degrees)
        self.Wx, self.Ws, self.bs, self.Wm, self.bm, self.Wp, self.bp = create_weights_conditional(n_conditionals,
                                                                                                   n_inputs,
                                                                                                   n_hiddens, None)
        self.input_order = self.degrees[0]

        self.activation_function = get_activation_function(activation)

        # Output info. TODO: make these not properties of self
        self.m = None
        self.logp = None

        # Dtype and GPU / CPU management
        self.to_args = None
        self.to_kwargs = None

    def forward(self, theta, x, **kwargs):
        """

        Parameters
        ----------
        theta :
            
        x :
            
        **kwargs :
            

        Returns
        -------

        """
        # Conditioner
        try:
            h = self.activation_function(
                F.linear(theta, torch.t(self.Wx)) + F.linear(x, torch.t(self.Ms[0] * self.Ws[0]), self.bs[0]))
        except RuntimeError:
            logging.error('Abort! Abort!')
            logging.info('MADE settings: n_inputs = %s, n_conditionals = %s', self.n_inputs, self.n_conditionals)
            logging.info('Shapes: theta %s, Wx %s, x %s, Ms %s, Ws %s, bs %s',
                         theta.shape, self.Wx.shape, x.shape, self.Ms[0].shape, self.Ws[0].shape, self.bs[0].shape)
            logging.info('Types: theta %s, Wx %s, x %s, Ms %s, Ws %s, bs %s',
                         type(theta), type(self.Wx), type(x), type(self.Ms[0]), type(self.Ws[0]), type(self.bs[0]))
            logging.info('CUDA: theta %s, Wx %s, x %s, Ms %s, Ws %s, bs %s',
                         theta.is_cuda, self.Wx.is_cuda, x.is_cuda, self.Ms[0].is_cuda, self.Ws[0].is_cuda,
                         self.bs[0].is_cuda)
            raise

        for M, W, b in zip(self.Ms[1:], self.Ws[1:], self.bs[1:]):
            h = self.activation_function(F.linear(h, torch.t(M * W), b))

        # Gaussian parameters
        self.m = F.linear(h, torch.t(self.Mmp * self.Wm), self.bm)
        self.logp = F.linear(h, torch.t(self.Mmp * self.Wp), self.bp)

        # u(x)
        u = torch.exp(0.5 * self.logp) * (x - self.m)

        # log det du/dx
        logdet_dudx = 0.5 * torch.sum(self.logp, dim=1)

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

        x = torch.zeros([n_samples, self.n_inputs])
        if u is None:
            u = tensor(rng.randn(n_samples, self.n_inputs))

        if self.to_args is not None or self.to_kwargs is not None:
            x = x.to(*self.to_args, **self.to_kwargs)
            u = u.to(*self.to_args, **self.to_kwargs)

        for i in range(1, self.n_inputs + 1):
            self.forward(theta, x)  # Sets Gaussian parameters: self.m and self.logp

            idx = np.argwhere(self.input_order == i)[0, 0]

            mask = torch.zeros([n_samples, self.n_inputs])
            if self.to_args is not None or self.to_kwargs is not None:
                mask = mask.to(*self.to_args, **self.to_kwargs)

            mask[:, idx] = 1.

            x = (1. - mask) * x + mask * (self.m + torch.exp(torch.clamp(-0.5 * self.logp, -10., 10.)) * u)

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

        for i, (M, W, b) in enumerate(zip(self.Ms, self.Ws, self.bs)):
            self.Ms[i] = M.to(*args, **kwargs)
            self.Ws[i] = nn.Parameter(W.to(*args, **kwargs))
            self.bs[i] = nn.Parameter(b.to(*args, **kwargs))

        self.Mmp = self.Mmp.to(*args, **kwargs)
        self.Wx = nn.Parameter(self.Wx.to(*args, **kwargs))
        self.Wm = nn.Parameter(self.Wm.to(*args, **kwargs))
        self.bm = nn.Parameter(self.bm.to(*args, **kwargs))
        self.Wp = nn.Parameter(self.Wp.to(*args, **kwargs))
        self.bp = nn.Parameter(self.bp.to(*args, **kwargs))

        return self
