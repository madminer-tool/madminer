from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import six
import logging

logger = logging.getLogger(__name__)


def _parse_theta(theta, n_samples):
    theta_type_in = theta[0]
    theta_value_in = theta[1]

    if theta_type_in == "benchmark":
        thetas_out = [int(theta_value_in)]
        n_samples_per_theta = n_samples

    elif theta_type_in == "benchmarks":
        n_benchmarks = len(theta_value_in)
        n_samples_per_theta = int(round(n_samples / n_benchmarks, 0))
        thetas_out = [int(val) for val in theta_value_in]

    elif theta_type_in == "theta":
        thetas_out = np.asarray([theta_value_in])
        n_samples_per_theta = n_samples

    elif theta_type_in == "thetas":
        n_benchmarks = len(theta_value_in)
        n_samples_per_theta = int(round(n_samples / n_benchmarks, 0))
        thetas_out = np.asarray(theta_value_in)

    elif theta_type_in == "random":
        n_benchmarks, priors = theta_value_in
        if n_benchmarks is None or n_benchmarks <= 0:
            n_benchmarks = n_samples
        n_samples_per_theta = int(round(n_samples / n_benchmarks, 0))

        thetas_out = []
        for prior in priors:
            if prior[0] == "flat":
                prior_min = prior[1]
                prior_max = prior[2]
                thetas_out.append(prior_min + (prior_max - prior_min) * np.random.rand(n_benchmarks))
            elif prior[0] == "gaussian":
                prior_mean = prior[1]
                prior_std = prior[2]
                thetas_out.append(np.random.normal(loc=prior_mean, scale=prior_std, size=n_benchmarks))
            else:
                raise ValueError("Unknown prior {}".format(prior))
        thetas_out = np.array(thetas_out).T

    else:
        raise ValueError("Unknown theta specification {}".format(theta))

    return thetas_out, n_samples_per_theta


def _get_theta_value(theta, benchmarks):
    if isinstance(theta, six.string_types):
        benchmark = benchmarks[theta]
        theta_value = np.array([benchmark[key] for key in benchmark])
    elif isinstance(theta, int):
        benchmark = benchmarks[list(benchmarks.keys())[theta]]
        theta_value = np.array([benchmark[key] for key in benchmark])
    else:
        theta_value = np.asarray(theta)
    return theta_value


def _get_nu_value(nu, benchmarks):
    if isinstance(nu, None):
        nu_value = 0.0
    else:
        nu_value = np.asarray(nu)
    return nu_value


def _get_theta_benchmark_matrix(theta, benchmarks, morpher=None):
    """Calculates vector A such that dsigma(theta) = A * dsigma_benchmarks"""

    if isinstance(theta, six.string_types):
        i_benchmark = list(benchmarks).index(theta)
        return _get_theta_benchmark_matrix(i_benchmark, benchmarks, morpher)
    elif isinstance(theta, int):
        n_benchmarks = len(benchmarks)
        theta_matrix = np.zeros(n_benchmarks)
        theta_matrix[theta] = 1.0
    else:
        theta_matrix = morpher.calculate_morphing_weights(theta)
    return theta_matrix


def _get_dtheta_benchmark_matrix(theta, benchmarks, morpher):
    """Calculates matrix A_ij such that d dsigma(theta) / d theta_i = A_ij * dsigma (benchmark j)"""

    if morpher is None:
        raise RuntimeError("Cannot calculate score without morphing")

    if isinstance(theta, six.string_types):
        benchmark = benchmarks[theta]
        benchmark = np.array([value for _, value in six.iteritems(benchmark)])
        return _get_dtheta_benchmark_matrix(benchmark, benchmarks, morpher)

    elif isinstance(theta, int):
        benchmark = benchmarks[list(benchmarks.keys())[theta]]
        benchmark = np.array([value for _, value in six.iteritems(benchmark)])
        return _get_dtheta_benchmark_matrix(benchmark, benchmarks, morpher)

    else:
        dtheta_matrix = morpher.calculate_morphing_weight_gradient(theta)  # Shape (n_parameters, n_benchmarks_phys)

    return dtheta_matrix


def _calculate_augmented_data(
    augmented_data_definitions,
    weights_benchmarks,
    xsecs_benchmarks,
    theta_matrices,
    theta_gradient_matrices,
    nuisance_morpher=None,
):
    """Extracts augmented data from benchmark weights"""

    augmented_data = []

    for definition in augmented_data_definitions:

        if definition[0] == "ratio":
            i_num = definition[1]
            i_den = definition[2]

            dsigma_num = mdot(theta_matrices[i_num], weights_benchmarks)
            sigma_num = mdot(theta_matrices[i_num], xsecs_benchmarks)
            dsigma_den = mdot(theta_matrices[i_den], weights_benchmarks)
            sigma_den = mdot(theta_matrices[i_den], xsecs_benchmarks)

            ratio = (dsigma_num / sigma_num) / (dsigma_den / sigma_den)
            ratio = ratio.reshape((-1, 1))

            augmented_data.append(ratio)

        elif definition[0] == "score":
            i = definition[1]

            gradient_dsigma = mdot(theta_gradient_matrices[i], weights_benchmarks)  # (n_gradients, n_samples)
            gradient_sigma = mdot(theta_gradient_matrices[i], xsecs_benchmarks)  # (n_gradients,)

            dsigma = mdot(theta_matrices[i], weights_benchmarks)  # (n_samples,)
            sigma = mdot(theta_matrices[i], xsecs_benchmarks)  # scalar

            score = gradient_dsigma / dsigma  # (n_gradients, n_samples)
            score = score.T  # (n_samples, n_gradients)
            score = score - np.broadcast_to(gradient_sigma / sigma, score.shape)  # (n_samples, n_gradients)

            augmented_data.append(score)

        elif definition[0] == "nuisance_score":
            a_weights = nuisance_morpher.calculate_a(weights_benchmarks)
            a_xsec = nuisance_morpher.calculate_a(xsecs_benchmarks[np.newaxis, :])

            nuisance_score = a_weights - a_xsec  # Shape (n_nuisance_parameters, n_samples)
            nuisance_score = nuisance_score.T  # Shape (n_samples, n_nuisance_parameters)

            logger.debug("Nuisance score: shape %s, content %s", nuisance_score.shape, nuisance_score)

            augmented_data.append(nuisance_score)

        else:
            raise ValueError("Unknown augmented data type {}".format(definition[0]))

    return augmented_data


def mdot(matrix, benchmark_information):
    """
    Calculates a product between a matrix / matrices with shape (n1) or (a, n1) and a weight list with shape (b, n2)
    or (n2,), where n1 and n2 do not have to be the same
    """

    n_benchmarks_matrix = matrix.shape[-1]
    weights_benchmarks_T = benchmark_information.T
    n_benchmarks_list = weights_benchmarks_T.shape[0]
    n_smaller = min(n_benchmarks_matrix, n_benchmarks_list)

    if n_benchmarks_matrix == n_benchmarks_list:
        return matrix.dot(weights_benchmarks_T)

    if n_benchmarks_matrix < n_benchmarks_list:
        matrix = matrix.T
        matrix = matrix[:n_smaller]
        matrix = matrix.T

    return matrix.dot(weights_benchmarks_T[:n_smaller])
