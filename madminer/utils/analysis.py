from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import six
import logging


def get_theta_value(theta_type, theta_value, benchmarks):
    if theta_type == "benchmark":
        benchmark = benchmarks[theta_value]
        benchmark_theta = np.array([benchmark[key] for key in benchmark])
        return benchmark_theta

    elif theta_type == "morphing":
        return theta_value

    else:
        raise ValueError("Unknown theta {}".format(theta_type))


def get_theta_benchmark_matrix(theta_type, theta_value, benchmarks, morpher=None):
    """Calculates vector A such that dsigma(theta) = A * dsigma_benchmarks"""

    if theta_type == "benchmark":
        n_benchmarks = len(benchmarks)
        index = list(benchmarks).index(theta_value)
        theta_matrix = np.zeros(n_benchmarks)
        theta_matrix[index] = 1.0

    elif theta_type == "morphing":
        theta_matrix = morpher.calculate_morphing_weights(theta_value)

    else:
        raise ValueError("Unknown theta {}".format(theta_type))

    return theta_matrix


def get_dtheta_benchmark_matrix(theta_type, theta_value, benchmarks, morpher=None):
    """Calculates matrix A_ij such that d dsigma(theta) / d theta_i = A_ij * dsigma (benchmark j)"""

    if theta_type == "benchmark":
        if morpher is None:
            raise RuntimeError("Cannot calculate score without morphing")

        theta = benchmarks[theta_value]
        theta = np.array([value for _, value in six.iteritems(theta)])

        return get_dtheta_benchmark_matrix("morphing", theta, benchmarks, morpher)

    elif theta_type == "morphing":
        if morpher is None:
            raise RuntimeError("Cannot calculate score without morphing")

        dtheta_matrix = morpher.calculate_morphing_weight_gradient(theta_value)  # Shape (n_parameters, n_benchmarks)

    else:
        raise ValueError("Unknown theta {}".format(theta_type))

    return dtheta_matrix


def extract_augmented_data(
    augmented_data_definitions, weights_benchmarks, xsecs_benchmarks, theta_matrices, theta_gradient_matrices
):
    """Extracts augmented data from benchmark weights"""
    augmented_data = []

    for definition in augmented_data_definitions:

        if definition[0] == "ratio":
            i_num = definition[1]
            i_den = definition[2]

            dsigma_num = theta_matrices[i_num].dot(weights_benchmarks.T)
            sigma_num = theta_matrices[i_num].dot(xsecs_benchmarks.T)
            dsigma_den = theta_matrices[i_den].dot(weights_benchmarks.T)
            sigma_den = theta_matrices[i_den].dot(xsecs_benchmarks.T)

            ratio = (dsigma_num / sigma_num) / (dsigma_den / sigma_den)
            ratio = ratio.reshape((-1, 1))

            augmented_data.append(ratio)

        elif definition[0] == "score":
            i = definition[1]

            gradient_dsigma = theta_gradient_matrices[i].dot(weights_benchmarks.T)  # (n_gradients, n_samples)
            gradient_sigma = theta_gradient_matrices[i].dot(xsecs_benchmarks.T)  # (n_gradients,)

            dsigma = theta_matrices[i].dot(weights_benchmarks.T)  # (n_samples,)
            sigma = theta_matrices[i].dot(xsecs_benchmarks.T)  # scalar

            score = gradient_dsigma / dsigma  # (n_gradients, n_samples)
            score = score.T  # (n_samples, n_gradients)
            score = score - np.broadcast_to(gradient_sigma / sigma, score.shape)  # (n_samples, n_gradients)

            augmented_data.append(score)

        else:
            raise ValueError("Unknown augmented data type {}".format(definition[0]))

    return augmented_data


def parse_theta(theta, n_samples):
    theta_type_in = theta[0]
    theta_value_in = theta[1]

    if theta_type_in == "benchmark":
        theta_types = ["benchmark"]
        theta_values = [theta_value_in]
        n_samples_per_theta = n_samples

    elif theta_type_in == "benchmarks":
        n_benchmarks = len(theta_value_in)
        theta_types = ["benchmark"] * n_benchmarks
        theta_values = theta_value_in
        n_samples_per_theta = int(round(n_samples / n_benchmarks, 0))

    elif theta_type_in == "theta":
        theta_types = ["morphing"]
        theta_values = [theta_value_in]
        n_samples_per_theta = n_samples

    elif theta_type_in == "thetas":
        n_benchmarks = len(theta_value_in)
        theta_types = ["morphing"] * n_benchmarks
        theta_values = theta_value_in
        n_samples_per_theta = int(round(n_samples / n_benchmarks, 0))

    elif theta_type_in == "random":
        n_benchmarks, priors = theta_value_in

        if n_benchmarks is None or n_benchmarks <= 0:
            n_benchmarks = n_samples

        theta_values = []
        for prior in priors:
            if prior[0] == "flat":
                prior_min = prior[1]
                prior_max = prior[2]
                theta_values.append(prior_min + (prior_max - prior_min) * np.random.rand(n_benchmarks))

            elif prior[0] == "gaussian":
                prior_mean = prior[1]
                prior_std = prior[2]
                theta_values.append(np.random.normal(loc=prior_mean, scale=prior_std, size=n_benchmarks))

            else:
                raise ValueError("Unknown prior {}".format(prior))

        theta_types = ["morphing"] * n_benchmarks
        theta_values = np.array(theta_values).T
        n_samples_per_theta = int(round(n_samples / n_benchmarks, 0))

        logging.debug(
            "Total n_samples: %s, n_benchmarks: %s, n_samples_per_theta: %s",
            n_samples,
            n_benchmarks,
            n_samples_per_theta,
        )

    else:
        raise ValueError("Unknown theta {}".format(theta))

    return theta_types, theta_values, n_samples_per_theta
