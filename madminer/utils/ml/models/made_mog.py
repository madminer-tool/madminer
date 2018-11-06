from __future__ import absolute_import, division, print_function

import numpy as np
import numpy.random as rng
import logging

import torch
from torch import tensor
import torch.nn.functional as F

from madminer.utils.ml.models.base import BaseConditionalFlow
from madminer.utils.ml.models.masks import create_degrees, create_masks, create_weights_conditional
from madminer.utils.ml.utils import get_activation_function


class ConditionalMixtureMADE(BaseConditionalFlow):
    """ """
    def __init__(self, n_conditionals, n_inputs, n_hiddens, n_components=10, activation='relu',
                 input_order='sequential',
                 mode='sequential'):
        super(ConditionalMixtureMADE, self).__init__(n_conditionals, n_inputs)

        # save input arguments
        self.activation = activation
        self.n_conditionals = n_conditionals
        self.n_inputs = n_inputs
        self.n_hiddens = n_hiddens
        self.mode = mode
        self.n_components = n_components

        # create network's parameters
        self.degrees = create_degrees(n_inputs, n_hiddens, input_order, mode)
        self.Ms, self.Mmp = create_masks(self.degrees)
        logging.debug('Mmp shape: %s', self.Mmp.shape)
        (self.Wx, self.Ws, self.bs, self.Wm, self.bm, self.Wp, self.bp, self.Wa,
         self.ba) = create_weights_conditional(
            n_conditionals,
            n_inputs,
            n_hiddens,
            n_components
        )
        self.input_order = self.degrees[0]

        # Shaping things
        self.Mmp = self.Mmp.unsqueeze(2)
        self.ba.data = self.ba.data.unsqueeze(0)
        self.bp.data = self.bp.data.unsqueeze(0)
        self.bm.data = self.bm.data.unsqueeze(0)

        self.activation_function = get_activation_function(activation)

        # Output info. TODO: make these not properties of self
        self.m = None
        self.logp = None
        self.loga = None

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
            logging.info('dtypes: theta %s, Wx %s, x %s, Ms %s, Ws %s, bs %s',
                         theta.dtype, self.Wx.dtype, x.dtype, self.Ms[0].dtype, self.Ws[0].dtype, self.bs[0].dtype)
            logging.info('Types: theta %s, Wx %s, x %s, Ms %s, Ws %s, bs %s',
                         type(theta), type(self.Wx), type(x), type(self.Ms[0]), type(self.Ws[0]), type(self.bs[0]))
            logging.info('CUDA: theta %s, Wx %s, x %s, Ms %s, Ws %s, bs %s',
                         theta.is_cuda, self.Wx.is_cuda, x.is_cuda, self.Ms[0].is_cuda, self.Ws[0].is_cuda,
                         self.bs[0].is_cuda)
            raise

        for M, W, b in zip(self.Ms[1:], self.Ws[1:], self.bs[1:]):
            h = self.activation_function(F.linear(h, torch.t(M * W), b))

        # Gaussian mean
        h = h.unsqueeze(1).unsqueeze(2)
        weight = (self.Mmp * self.Wm).transpose(0, 1).contiguous().unsqueeze(0)
        self.m = torch.matmul(h, weight).squeeze() + self.bm

        # Gaussian variance
        weight = (self.Mmp * self.Wp).transpose(0, 1).contiguous().unsqueeze(0)
        self.logp = torch.matmul(h, weight).squeeze() + self.bp

        # mixing coefficients
        weight = (self.Mmp * self.Wa).transpose(0, 1).contiguous().unsqueeze(0)
        self.loga = torch.matmul(h, weight).squeeze() + self.ba
        self.loga -= torch.log(torch.sum(torch.exp(self.loga), dim=2, keepdim=True))

        # u(x)
        x = x.unsqueeze(2)
        u = torch.exp(0.5 * self.logp) * (x - self.m)

        # log det du/dx
        logdet_dudx = 0.5 * self.logp  # torch.sum(self.logp, dim=1)

        return u, logdet_dudx, self.loga

    def log_likelihood(self, theta, x, **kwargs):
        """Calculates u(x) and log p(x) with a Gaussian base density"""

        u, logdet_dudx, log_a = self.forward(theta, x, **kwargs)

        constant = float(- 0.5 * self.n_inputs * np.log(2. * np.pi))
        # log_likelihood = torch.log(torch.sum(torch.exp(log_a - 0.5 * u ** 2 + logdet_dudx), dim=2))
        log_likelihood = torch.logsumexp(log_a - 0.5 * u ** 2 + logdet_dudx, dim=2)
        log_likelihood = constant + torch.sum(log_likelihood, dim=1)

        return u, log_likelihood

    def generate_samples(self, theta, u=None, **kwargs):
        n_samples = theta.shape[0]

        x = torch.zeros([n_samples, self.n_inputs])
        if u is None:
            u = tensor(rng.randn(n_samples, self.n_inputs))

        if self.to_args is not None or self.to_kwargs is not None:
            x = x.to(*self.to_args, **self.to_kwargs)
            u = u.to(*self.to_args, **self.to_kwargs)

        for i in range(1, self.n_inputs + 1):
            self.forward(theta, x)  # Sets Gaussian parameters (self.m and self.logp) and mixture coeffs (self.loga)

            ix = np.argwhere(self.input_order == i)[0, 0]

            for i_sample in range(n_samples):
                # Mask
                mask = torch.zeros([n_samples, self.n_inputs])
                if self.to_args is not None or self.to_kwargs is not None:
                    mask = mask.to(*self.to_args, **self.to_kwargs)
                mask[i_sample, ix] = 1.

                # Mixture component
                p_components = np.exp(self.loga[i_sample, ix].detach().numpy())
                cum_p_components = np.cumsum(p_components[:-1])
                r = np.random.rand(1)
                c = np.sum((r > cum_p_components).astype(int))

                x = ((1. - mask) * x
                     + mask * (self.m[:, :, c] + torch.exp(torch.clamp(-0.5 * self.logp[:, :, c], -10., 10.)) * u))

        return x

    def to(self, *args, **kwargs):
        self.to_args = args
        self.to_kwargs = kwargs

        self = super().to(*args, **kwargs)

        for i, M in enumerate(self.Ms):
            self.Ms[i] = M.to(*args, **kwargs)
        self.Mmp = self.Mmp.to(*args, **kwargs)

        return self
