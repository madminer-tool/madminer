import numpy as np
import numpy.random as rng
import logging

import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F

from .masks import create_degrees, create_masks, create_weights, create_weights_conditional
from goldmine.various.utils import get_activation_function


class GaussianMADE(nn.Module):

    def __init__(self, n_inputs, n_hiddens, activation='relu', input_order='sequential', mode='sequential'):

        """
        Constructor.
        :param n_inputs: number of inputs
        :param n_hiddens: list with number of hidden units for each hidden layer
        :param input_order: order of inputs
        :param mode: strategy for assigning degrees to hidden nodes: can be 'random' or 'sequential'
        """

        super(GaussianMADE, self).__init__()

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
        self.log_likelihood = None

    def forward(self, x):

        """ Transforms x into u = f^-1(x) """

        h = x

        # feedforward propagation
        for M, W, b in zip(self.Ms, self.Ws, self.bs):
            h = self.activation_function(F.linear(h, torch.t(M * W), b))

        # output means
        self.m = F.linear(h, torch.t(self.Mmp * self.Wm), self.bm)

        # output log precisions
        self.logp = F.linear(h, torch.t(self.Mmp * self.Wp), self.bp)

        # random numbers driving made
        u = torch.exp(0.5 * self.logp) * (x - self.m)

        # log likelihoods
        diff = torch.sum(u ** 2 - self.logp, dim=1)
        constant = float(self.n_inputs * np.log(2. * np.pi))
        self.log_likelihood = -0.5 * (constant + diff)

        return u

    def log_p(self, x):

        """ Calculates log p(x) """

        _ = self.forward(x)

        return self.log_likelihood

    def gen(self, n_samples=1, u=None):
        """
        Generate samples from made. Requires as many evaluations as number of inputs.
        :param n_samples: number of samples
        :param u: random numbers to use in generating samples; if None, new random numbers are drawn
        :return: samples
        """

        # TODO: reformulate in pyTorch instread of numpy

        x = np.zeros([n_samples, self.n_inputs])
        u = rng.randn(n_samples, self.n_inputs) if u is None else u.data.numpy()

        for i in range(1, self.n_inputs + 1):
            self.forward(tensor(x))  # Sets Gaussian parameters: self.m and self.logp
            m = self.m.data.numpy()
            logp = self.logp.data.numpy()

            idx = np.argwhere(self.input_order == i)[0, 0]
            x[:, idx] = m[:, idx] + np.exp(np.minimum(-0.5 * logp[:, idx], 10.0)) * u[:, idx]

        return tensor(x)


class ConditionalGaussianMADE(nn.Module):
    """
    Implements a Made, where each conditional probability is modelled by a single gaussian component. The made has
    inputs theta which is always conditioned on, and whose probability it doesn't model.
    """

    def __init__(self, n_conditionals, n_inputs, n_hiddens, activation='relu', input_order='sequential',
                 mode='sequential'):
        """
        Constructor.
        :param n_conditionals: number of (conditional) inputs theta
        :param n_inputs: number of inputs X
        :param n_hiddens: list with number of hidden units for each hidden layer
        :param output_order: order of outputs
        :param mode: strategy for assigning degrees to hidden nodes: can be 'random' or 'sequential'
        """

        super(ConditionalGaussianMADE, self).__init__()

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

        # Output info
        self.m = None
        self.logp = None
        self.log_likelihood = None

    def forward(self, theta, x):

        """ Transforms theta, x into u = f^-1(x | theta) """

        # First hidden layer

        # Debug

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
                         theta.is_cuda, self.Wx.is_cuda, x.is_cuda, self.Ms[0].is_cuda, self.Ws[0].is_cuda, self.bs[0].is_cuda)
            raise

        # feedforward propagation
        for M, W, b in zip(self.Ms[1:], self.Ws[1:], self.bs[1:]):
            h = self.activation_function(F.linear(h, torch.t(M * W), b))

        # output means
        self.m = F.linear(h, torch.t(self.Mmp * self.Wm), self.bm)

        # output log precisions
        self.logp = F.linear(h, torch.t(self.Mmp * self.Wp), self.bp)

        # random numbers driving made
        u = torch.exp(0.5 * self.logp) * (x - self.m)

        # log likelihoods
        diff = torch.sum(u ** 2 - self.logp, dim=1)
        constant = float(self.n_inputs * np.log(2. * np.pi))
        self.log_likelihood = -0.5 * (constant + diff)

        return u

    def predict_log_likelihood(self, theta, x):

        """ Calculates log p(x) """

        _ = self.forward(theta, x)

        return self.log_likelihood

    def generate_samples(self, theta, u=None):
        """
        Generate samples from made. Requires as many evaluations as number of inputs.
        :param theta: conditionals
        :param n_samples: number of samples
        :param u: random numbers to use in generating samples; if None, new random numbers are drawn
        :return: samples
        """

        # TODO: reformulate in pyTorch instread of numpy

        n_samples = theta.shape[0]

        x = np.zeros([n_samples, self.n_inputs])
        u = rng.randn(n_samples, self.n_inputs) if u is None else u.data.numpy()

        for i in range(1, self.n_inputs + 1):
            self.forward(tensor(theta), tensor(x))
            m = self.m.data.numpy()
            logp = self.logp.data.numpy()

            idx = np.argwhere(self.input_order == i)[0, 0]
            x[:, idx] = m[:, idx] + np.exp(np.minimum(-0.5 * logp[:, idx], 10.0)) * u[:, idx]

        return tensor(x)

    def to(self, *args, **kwargs):

        logging.debug('Transforming MADE to %s', args)

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
