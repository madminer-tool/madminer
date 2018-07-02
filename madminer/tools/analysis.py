from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np


def get_theta_value(theta, benchmarks):
    if theta[0] == 'benchmark':
        benchmark = benchmarks[theta[1]]
        benchmark_theta = np.array([benchmark[key] for key in benchmark])
        return benchmark_theta

    elif theta[0] == 'morphing':
        return theta[1]

    else:
        raise ValueError('Unknown theta {}'.format(theta))


def get_theta_benchmark_matrix(theta, n_benchmarks, morpher=None):
    """ Calculates vector A such that dsigma(theta) = A * dsigma_benchmarks  """

    if theta[0] == 'benchmark':
        theta_matrix = np.zeros(n_benchmarks)
        theta_matrix[theta[1]] = 1.

    elif theta[1] == 'morphing':
        raise NotImplementedError

    else:
        raise ValueError('Unknown theta {}'.format(theta))


def get_dtheta_benchmark_matrix(theta, n_benchmarks, morpher=None):
    """ Calculates matrix A_ij such that d dsigma(theta) / d theta_i = A_ij * dsigma (benchmark j)  """

    if theta[0] == 'benchmark':
        theta_matrix = np.zeros(n_benchmarks)
        theta_matrix[theta[1]] = 1.

    elif theta[1] == 'morphing':
        raise NotImplementedError

    else:
        raise ValueError('Unknown theta {}'.format(theta))


def extract_augmented_data(types,
                           thetas_matrix_num,
                           thetas_matrix_den,
                           weights_benchmarks,
                           xsecs_benchmarks):
    n_augmented_data = len(thetas_matrix_num)
    n_samples = weights_benchmarks.shape[0]

    augmented_data = []

    for type, theta_matrix_num, theta_matrix_den in zip(types, thetas_matrix_num, thetas_matrix_den):

        # Numerator of ratio / d_i p(x|theta) for score
        dsigma_num = theta_matrix_num.dot(weights_benchmarks)
        sigma_num = theta_matrix_num.dot(xsecs_benchmarks)

        # Denominator of ratio / p(x|theta) for score
        dsigma_den = theta_matrix_den.dot(weights_benchmarks)
        sigma_den = theta_matrix_den.dot(xsecs_benchmarks)

        # Calculate ratio
        if type == 'ratio':
            augmented_datum = (dsigma_num / sigma_num) / (dsigma_den / sigma_num)

        # Calculate score
        elif type == 'score':
            augmented_datum = (dsigma_num / dsigma_den) - (sigma_num / sigma_den)

        else:
            raise ValueError("Unknown augmented data type {}", type)

        augmented_data.append(augmented_datum)

    augmented_data = np.array(augmented_data).T
    return augmented_data
