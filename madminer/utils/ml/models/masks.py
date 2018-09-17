import numpy as np
import numpy.random as rng

from torch import tensor
import torch.nn as nn

# TODO: switch to pure torch syntax
# TODO: consider unifying dtypes (float?)


def create_degrees(n_inputs, n_hiddens, input_order, mode):
    """
    Generates a degree for each hidden and input unit. A unit with degree d can only receive input from units with
    degree less than d.
    :param n_inputs: the number of inputs
    :param n_hiddens: a list with the number of hidden units
    :param input_order: the order of the inputs; can be 'random', 'sequential', or an array of an explicit order
    :param mode: the strategy for assigning degrees to hidden nodes: can be 'random' or 'sequential'
    :return: list of degrees
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
        for N in n_hiddens:
            min_prev_degree = min(np.min(degrees[-1]), n_inputs - 1)
            degrees_l = rng.randint(min_prev_degree, n_inputs, N)
            degrees.append(degrees_l)

    elif mode == 'sequential':
        for N in n_hiddens:
            degrees_l = np.arange(N) % max(1, n_inputs - 1) + min(1, n_inputs - 1)
            degrees.append(degrees_l)

    else:
        raise ValueError('invalid mode')

    return degrees


def create_masks(degrees):
    """
    Creates the binary masks that make the connectivity autoregressive.
    :param degrees: a list of degrees for every layer
    :return: list of all masks, as theano shared variables
    """

    Ms = []

    for l, (d0, d1) in enumerate(zip(degrees[:-1], degrees[1:])):
        M = (d0[:, np.newaxis] <= d1).astype(np.float)
        M = tensor(M)
        Ms.append(M)

    Mmp = (degrees[-1][:, np.newaxis] < degrees[0]).astype(np.float)
    Mmp = tensor(Mmp)

    return Ms, Mmp


def create_weights(n_inputs, n_hiddens, n_comps=None):
    """
    Creates all learnable weight matrices and bias vectors.
    :param n_inputs: the number of inputs
    :param n_hiddens: a list with the number of hidden units
    :param n_comps: number of gaussian components
    :return: weights and biases, as theano shared variables
    """

    Ws = []
    bs = []

    n_units = np.concatenate(([n_inputs], n_hiddens))

    for N0, N1 in zip(n_units[:-1], n_units[1:]):
        W = nn.Parameter(tensor((rng.randn(N0, N1) / np.sqrt(N0 + 1))))
        b = nn.Parameter(tensor(np.zeros((N1,))))
        Ws.append(W)
        bs.append(b)

    if n_comps is None:
        Wm = nn.Parameter(tensor((rng.randn(n_units[-1], n_inputs) / np.sqrt(n_units[-1] + 1))))
        Wp = nn.Parameter(tensor((rng.randn(n_units[-1], n_inputs) / np.sqrt(n_units[-1] + 1))))
        bm = nn.Parameter(tensor(np.zeros((n_inputs,))))
        bp = nn.Parameter(tensor(np.zeros((n_inputs,))))

        return Ws, bs, Wm, bm, Wp, bp
    else:

        Wm = nn.Parameter(tensor((rng.randn(n_units[-1], n_inputs, n_comps) / np.sqrt(n_units[-1] + 1))))
        Wp = nn.Parameter(tensor((rng.randn(n_units[-1], n_inputs, n_comps) / np.sqrt(n_units[-1] + 1))))
        Wa = nn.Parameter(tensor((rng.randn(n_units[-1], n_inputs, n_comps) / np.sqrt(n_units[-1] + 1))))
        bm = nn.Parameter(tensor(rng.randn(n_inputs, n_comps)))
        bp = nn.Parameter(tensor(rng.randn(n_inputs, n_comps)))
        ba = nn.Parameter(tensor(rng.randn(n_inputs, n_comps)))

        return Ws, bs, Wm, bm, Wp, bp, Wa, ba


def create_weights_conditional(n_conditionals, n_inputs, n_hiddens, n_comps):
    """
    Creates all learnable weight matrices and bias vectors for a conditional made.
    :param n_conditionals: the number of (conditional) inputs theta
    :param n_inputs: the number of unconditional inputs x
    :param n_hiddens: a list with the number of hidden units
    :param n_comps: number of gaussian components
    :return: weights and biases, as theano shared variables
    """

    Wx = nn.Parameter(tensor((rng.randn(n_conditionals, n_hiddens[0]) / np.sqrt(n_conditionals + 1))))
    return (Wx,) + create_weights(n_inputs, n_hiddens, n_comps)
