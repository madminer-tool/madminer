from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np


def get_theta_value(theta_type, theta_value, benchmarks):
    if theta_type == 'benchmark':
        benchmark = benchmarks[list(benchmarks.keys())[theta_value]]
        benchmark_theta = np.array([benchmark[key] for key in benchmark])
        return benchmark_theta

    elif theta_type == 'morphing':
        return theta_value

    else:
        raise ValueError('Unknown theta {}'.format(theta_type))


def get_theta_benchmark_matrix(theta_type, theta_value, n_benchmarks, morpher=None):
    """ Calculates vector A such that dsigma(theta) = A * dsigma_benchmarks  """

    if theta_type == 'benchmark':
        theta_matrix = np.zeros(n_benchmarks)
        theta_matrix[theta_value] = 1.

    elif theta_type == 'morphing':
        theta_matrix = morpher.calculate_morphing_weights(theta_value)

    else:
        raise ValueError('Unknown theta {}'.format(theta_type))

    return theta_matrix


def get_dtheta_benchmark_matrix(theta_type, theta_value, n_benchmarks, morpher=None):
    """ Calculates matrix A_ij such that d dsigma(theta) / d theta_i = A_ij * dsigma (benchmark j)  """

    # TODO

    if theta_type == 'benchmark':
        raise NotImplementedError

    elif theta_type == 'morphing':
        theta_matrix = morpher.calculate_morphing_weight_gradient(theta_value)

    else:
        raise ValueError('Unknown theta {}'.format(theta_type))


def extract_augmented_data(types,
                           theta_matrices_num,
                           theta_matrices_den,
                           weights_benchmarks,
                           xsecs_benchmarks):

    augmented_data = []

    for data_type, theta_matrix_num, theta_matrix_den in zip(types, theta_matrices_num, theta_matrices_den):

        # Numerator of ratio / d_i p(x|theta) for score
        dsigma_num = theta_matrix_num.dot(weights_benchmarks.T)
        sigma_num = theta_matrix_num.dot(xsecs_benchmarks.T)

        # Denominator of ratio / p(x|theta) for score
        dsigma_den = theta_matrix_den.dot(weights_benchmarks.T)
        sigma_den = theta_matrix_den.dot(xsecs_benchmarks.T)

        # Calculate ratio
        if data_type == 'ratio':
            augmented_datum = (dsigma_num / sigma_num) / (dsigma_den / sigma_num)

        # Calculate score
        elif data_type == 'score':
            augmented_datum = (dsigma_num / dsigma_den) - (sigma_num / sigma_den)

        else:
            raise ValueError("Unknown augmented data type {}", data_type)

        augmented_data.append(augmented_datum)

    augmented_data = np.array(augmented_data).T
    return augmented_data
