from __future__ import absolute_import, division, print_function

import numpy as np
import numpy.random as rng

from torch import tensor
import torch.nn as nn


def create_degrees(n_inputs, n_hiddens, input_order, mode):
    """Generates a degree for each hidden and input unit. A unit with degree d can only receive input from units with
    degree less than d.

    Parameters
    ----------
    n_inputs :
        the number of inputs
    n_hiddens :
        a list with the number of hidden units
    input_order :
        the order of the inputs; can be 'random', 'sequential', or an array of an explicit order
    mode :
        the strategy for assigning degrees to hidden nodes: can be 'random' or 'sequential'

    Returns
    -------
    type
        list of degrees

    """

    degrees = []

    # create degrees for inputs
    if isinstance(input_order, str):

        if input_order == 'random':
            degrees_0 = np.arange(1, n_inputs + 1)
            rng.shuffle(degrees_0)

        elif input_order == 'sequential':
            degrees_0 = np.arange(1, n_inputs + 1)

        else:
            raise ValueError('invalid input order')

    else:
        input_order = np.array(input_order)
        assert np.all(np.sort(input_order) == np.arange(1, n_inputs + 1)), 'invalid input order'
        degrees_0 = input_order
    degrees.append(degrees_0)

    # create degrees for hiddens
    if mode == 'random':
        for n in n_hiddens:
            min_prev_degree = min(np.min(degrees[-1]), n_inputs - 1)
            degrees_l = rng.randint(min_prev_degree, n_inputs, n)
            degrees.append(degrees_l)

    elif mode == 'sequential':
        for n in n_hiddens:
            degrees_l = np.arange(n) % max(1, n_inputs - 1) + min(1, n_inputs - 1)
            degrees.append(degrees_l)

    else:
        raise ValueError('invalid mode')

    return degrees


def create_masks(degrees):
    """Creates the binary masks that make the connectivity autoregressive.

    Parameters
    ----------
    degrees :
        a list of degrees for every layer

    Returns
    -------
    type
        list of all masks, as theano shared variables

    """

    ms = []

    for l, (d0, d1) in enumerate(zip(degrees[:-1], degrees[1:])):
        m = (d0[:, np.newaxis] <= d1).astype(np.float)
        m = tensor(m)
        ms.append(m)

    mmp = (degrees[-1][:, np.newaxis] < degrees[0]).astype(np.float)
    mmp = tensor(mmp)

    return ms, mmp


def create_weights(n_inputs, n_hiddens, n_comps=None):
    """Creates all learnable weight matrices and bias vectors.

    Parameters
    ----------
    n_inputs :
        the number of inputs
    n_hiddens :
        a list with the number of hidden units
    n_comps :
        number of gaussian components (Default value = None)

    Returns
    -------
    type
        weights and biases, as theano shared variables

    """

    ws = nn.ParameterList()
    bs = nn.ParameterList()

    n_units = np.concatenate(([n_inputs], n_hiddens))

    for n0, n1 in zip(n_units[:-1], n_units[1:]):
        w = nn.Parameter(tensor((rng.randn(n0, n1) / np.sqrt(n0 + 1))))
        b = nn.Parameter(tensor(np.zeros((n1,))))
        ws.append(w)
        bs.append(b)

    if n_comps is None:
        wm = nn.Parameter(tensor((rng.randn(n_units[-1], n_inputs) / np.sqrt(n_units[-1] + 1))))
        wp = nn.Parameter(tensor((rng.randn(n_units[-1], n_inputs) / np.sqrt(n_units[-1] + 1))))
        bm = nn.Parameter(tensor(np.zeros((n_inputs,))))
        bp = nn.Parameter(tensor(np.zeros((n_inputs,))))

        return ws, bs, wm, bm, wp, bp
    else:

        wm = nn.Parameter(tensor((rng.randn(n_units[-1], n_inputs, n_comps) / np.sqrt(n_units[-1] + 1))))
        wp = nn.Parameter(tensor((rng.randn(n_units[-1], n_inputs, n_comps) / np.sqrt(n_units[-1] + 1))))
        wa = nn.Parameter(tensor((rng.randn(n_units[-1], n_inputs, n_comps) / np.sqrt(n_units[-1] + 1))))
        bm = nn.Parameter(tensor(rng.randn(n_inputs, n_comps)))
        bp = nn.Parameter(tensor(rng.randn(n_inputs, n_comps)))
        ba = nn.Parameter(tensor(rng.randn(n_inputs, n_comps)))

        return ws, bs, wm, bm, wp, bp, wa, ba


def create_weights_conditional(n_conditionals, n_inputs, n_hiddens, n_comps):
    """Creates all learnable weight matrices and bias vectors for a conditional made.

    Parameters
    ----------
    n_conditionals :
        the number of (conditional) inputs theta
    n_inputs :
        the number of unconditional inputs x
    n_hiddens :
        a list with the number of hidden units
    n_comps :
        number of gaussian components

    Returns
    -------
    type
        weights and biases, as theano shared variables

    """

    wx = nn.Parameter(tensor((rng.randn(n_conditionals, n_hiddens[0]) / np.sqrt(n_conditionals + 1))))
    return (wx,) + create_weights(n_inputs, n_hiddens, n_comps)
