from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import six
import logging


def get_theta_value(theta_type, theta_value, benchmarks):
    if theta_type == 'benchmark':
        benchmark = benchmarks[theta_value]
        benchmark_theta = np.array([benchmark[key] for key in benchmark])
        return benchmark_theta

    elif theta_type == 'morphing':
        return theta_value

    else:
        raise ValueError('Unknown theta {}'.format(theta_type))


def get_theta_benchmark_matrix(theta_type, theta_value, benchmarks, morpher=None):
    """ Calculates vector A such that dsigma(theta) = A * dsigma_benchmarks  """

    if theta_type == 'benchmark':
        n_benchmarks = len(benchmarks)
        index = list(benchmarks).index(theta_value)
        theta_matrix = np.zeros(n_benchmarks)
        theta_matrix[index] = 1.

    elif theta_type == 'morphing':
        theta_matrix = morpher.calculate_morphing_weights(theta_value)

    elif theta_type == 'sampling':
        theta_matrix = 'sampling'

    elif theta_type == 'auxiliary':
        theta_matrix = 'auxiliary'

    else:
        raise ValueError('Unknown theta {}'.format(theta_type))

    return theta_matrix


def get_dtheta_benchmark_matrix(theta_type, theta_value, benchmarks, morpher=None):
    """ Calculates matrix A_ij such that d dsigma(theta) / d theta_i = A_ij * dsigma (benchmark j)  """

    if theta_type == 'benchmark':
        if morpher is None:
            raise RuntimeError("Cannot calculate score without morphing")

        theta = benchmarks[theta_value]
        theta = np.array([value for _, value in six.iteritems(theta)])

        return get_dtheta_benchmark_matrix('morphing', theta, benchmarks, morpher)

    elif theta_type == 'morphing':
        if morpher is None:
            raise RuntimeError("Cannot calculate score without morphing")

        dtheta_matrix = morpher.calculate_morphing_weight_gradient(theta_value)  # Shape (n_parameters, n_benchmarks)

    elif theta_type == 'sampling':
        dtheta_matrix = 'sampling_gradient'

    elif theta_type == 'auxiliary':
        dtheta_matrix = 'auxiliary_gradient'

    else:
        raise ValueError('Unknown theta {}'.format(theta_type))

    return dtheta_matrix


def extract_augmented_data(types,
                           theta_matrices_num,
                           theta_matrices_den,
                           weights_benchmarks,
                           xsecs_benchmarks,
                           theta_sampling_matrix,
                           theta_sampling_gradient_matrix,
                           theta_auxiliary_matrix,
                           theta_auxiliary_gradient_matrix):

    augmented_data = []

    for data_type, theta_matrix_num, theta_matrix_den in zip(types, theta_matrices_num, theta_matrices_den):

        # Dynamic numerator / denominators
        if theta_matrix_num == 'sampling':
            theta_matrix_num = theta_sampling_matrix
        elif theta_matrix_num == 'sampling_gradient':
            theta_matrix_num = theta_sampling_gradient_matrix
        elif theta_matrix_num == 'auxiliary':
            theta_matrix_num = theta_auxiliary_matrix
        elif theta_matrix_num == 'auxiliary_gradient':
            theta_matrix_num = theta_auxiliary_gradient_matrix

        if theta_matrix_den == 'sampling':
            theta_matrix_den = theta_sampling_matrix
        elif theta_matrix_den == 'sampling_gradient':
            raise ValueError(theta_matrix_den)
        elif theta_matrix_den == 'auxiliary':
            theta_matrix_den = theta_auxiliary_matrix
        elif theta_matrix_den == 'auxiliary_gradient':
            raise ValueError(theta_matrix_den)


        # Numerator of ratio / d_i p(x|theta) for score
        dsigma_num = theta_matrix_num.dot(weights_benchmarks.T)
        sigma_num = theta_matrix_num.dot(xsecs_benchmarks.T)

        # Denominator of ratio / p(x|theta) for score
        dsigma_den = theta_matrix_den.dot(weights_benchmarks.T)
        sigma_den = theta_matrix_den.dot(xsecs_benchmarks.T)

        # Shapes
        # theta_matrices_num: (n_benchmarks) or (n_gradients, n_benchmarks)
        # theta_matrices_den: (n_benchmarks)
        # weights_benchmarks: (n_samples, n_benchmarks)
        # xsecs_benchmarks:   (n_benchmarks)
        # dsigma_num: (n_samples) or (n_gradients, n_samples)
        # sigma_num: () or (n_gradients)
        # dsigma_den: (n_samples)
        # sigma_den: ()

        # Calculate ratio
        if data_type == 'ratio':
            augmented_datum = (dsigma_num / sigma_num) / (dsigma_den / sigma_den)
            augmented_datum = augmented_datum.reshape((-1,1))

            # augmented_datum: (n_samples, 1)

        # Calculate score
        elif data_type == 'score':
            augmented_datum = (dsigma_num / dsigma_den)  # (n_gradients, n_samples)
            augmented_datum = augmented_datum.T  # (n_samples, n_gradients)
            augmented_datum = augmented_datum - np.broadcast_to(sigma_num / sigma_den, augmented_datum.shape)

        else:
            raise ValueError("Unknown augmented data type {}", data_type)

        # Let's check shape
        augmented_data.append(augmented_datum)

    return augmented_data


def parse_theta(theta, n_samples):
    theta_type_in = theta[0]
    theta_value_in = theta[1]

    if theta_type_in == 'benchmark':
        theta_types = ['benchmark']
        theta_values = [theta_value_in]
        n_samples_per_theta = n_samples

    elif theta_type_in == 'benchmarks':
        n_benchmarks = len(theta_value_in)
        theta_types = ['benchmark'] * n_benchmarks
        theta_values = theta_value_in
        n_samples_per_theta = int(round(n_samples / n_benchmarks, 0))

    elif theta_type_in == 'theta':
        theta_types = ['morphing']
        theta_values = [theta_value_in]
        n_samples_per_theta = n_samples

    elif theta_type_in == 'thetas':
        n_benchmarks = len(theta_value_in)
        theta_types = ['morphing'] * n_benchmarks
        theta_values = theta_value_in
        n_samples_per_theta = int(round(n_samples / n_benchmarks, 0))

    elif theta_type_in == 'random':
        n_benchmarks, priors = theta_value_in

        if n_benchmarks is None or n_benchmarks <= 0:
            n_benchmarks = n_samples

        theta_values = []
        for prior in priors:
            if prior[0] == 'flat':
                prior_min = prior[1]
                prior_max = prior[2]
                theta_values.append(
                    prior_min + (prior_max - prior_min) * np.random.rand(n_benchmarks)
                )

            elif prior[0] == 'gaussian':
                prior_mean = prior[1]
                prior_std = prior[2]
                theta_values.append(
                    np.random.normal(loc=prior_mean, scale=prior_std, size=n_benchmarks)
                )

            else:
                raise ValueError('Unknown prior {}'.format(prior))

        theta_types = ['morphing'] * n_benchmarks
        theta_values = np.array(theta_values).T
        n_samples_per_theta = int(round(n_samples / n_benchmarks, 0))

    else:
        raise ValueError('Unknown theta {}'.format(theta))

    return theta_types, theta_values, n_samples_per_theta