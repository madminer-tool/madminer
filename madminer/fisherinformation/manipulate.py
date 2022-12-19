import logging
import numpy as np

from ..utils.various import separate_information_blocks

logger = logging.getLogger(__name__)


def project_information(fisher_information, remaining_components, covariance=None):
    """
    Calculates projections of a Fisher information matrix, that is, "deletes" the rows and columns corresponding to
    some parameters not of interest.

    Parameters
    ----------
    fisher_information : ndarray
        Original n x n Fisher information.

    remaining_components : list of int
        List with m entries, each an int with 0 <= remaining_components[i] < n. Denotes which parameters are kept, and
        their new order. All other parameters or projected out.

    covariance : ndarray or None, optional
        The covariance matrix of the original Fisher information with shape (n, n, n, n). If None, the error on the
        profiled information is not calculated. Default value: None.

    Returns
    -------
    projected_fisher_information : ndarray
        Projected m x m Fisher information, where the `i`-th row or column corresponds to the
        `remaining_components[i]`-th row or column of fisher_information.

    profiled_fisher_information_covariance : ndarray
        Covariance matrix of the projected Fisher information matrix with shape (m, m, m, m). Only returned if
        covariance is not None.

    """
    n_new = len(remaining_components)
    fisher_information_new = np.zeros([n_new, n_new])

    # Project information
    for xnew, xold in enumerate(remaining_components):
        for ynew, yold in enumerate(remaining_components):
            fisher_information_new[xnew, ynew] = fisher_information[xold, yold]

    # Project covariance matrix
    if covariance is not None:
        covariance_new = np.zeros([n_new, n_new, n_new, n_new])
        for xnew, xold in enumerate(remaining_components):
            for ynew, yold in enumerate(remaining_components):
                for znew, zold in enumerate(remaining_components):
                    for zznew, zzold in enumerate(remaining_components):
                        covariance_new[xnew, ynew, znew, zznew] = covariance[xold, yold, zold, zzold]

        return fisher_information_new, covariance_new

    return fisher_information_new


def profile_information(
    fisher_information,
    remaining_components,
    covariance=None,
    error_propagation_n_ensemble=1000,
    error_propagation_factor=1.0e-3,
):
    """
    Calculates the profiled Fisher information matrix as defined in Appendix A.4 of arXiv:1612.05261.

    Parameters
    ----------
    fisher_information : ndarray
        Original n x n Fisher information.

    remaining_components : list of int
        List with m entries, each an int with 0 <= remaining_components[i] < n. Denotes which parameters are kept, and
        their new order. All other parameters or profiled out.

    covariance : ndarray or None, optional
        The covariance matrix of the original Fisher information with shape (n, n, n, n). If None, the error on the
        profiled information is not calculated. Default value: None.

    error_propagation_n_ensemble : int, optional
        If covariance is not None, this sets the number of Fisher information matrices drawn from a normal distribution
        for the Monte-Carlo error propagation. Default value: 1000.

    error_propagation_factor : float, optional
        If covariance is not None, this factor multiplies the covariance of the distribution of Fisher information
        matrices. Smaller factors can avoid problems with ill-behaved Fisher information matrices. Default value: 1.e-3.

    Returns
    -------
    profiled_fisher_information : ndarray
        Profiled m x m Fisher information, where the `i`-th row or column corresponds to the
        `remaining_components[i]`-th row or column of fisher_information.

    profiled_fisher_information_covariance : ndarray
        Covariance matrix of the profiled Fisher information matrix with shape (m, m, m, m).

    """

    logger.debug("Profiling Fisher information")
    n_components = len(fisher_information)
    n_remaining_components = len(remaining_components)

    _, information_phys, information_mix, information_nuisance = separate_information_blocks(
        fisher_information, remaining_components
    )

    # Error propagation
    if covariance is not None:
        # Central value
        profiled_information = profile_information(
            fisher_information, remaining_components=remaining_components, covariance=None
        )

        # Draw toys
        information_toys = np.random.multivariate_normal(
            mean=fisher_information.reshape((-1,)),
            cov=error_propagation_factor * covariance.reshape(n_components**2, n_components**2),
            size=error_propagation_n_ensemble,
        )
        information_toys = information_toys.reshape(-1, n_components, n_components)

        # Profile each toy
        profiled_information_toys = np.array(
            [
                profile_information(info, remaining_components=remaining_components, covariance=None)
                for info in information_toys
            ]
        )

        # Calculate ensemble covariance
        toy_covariance = np.cov(profiled_information_toys.reshape(-1, n_remaining_components**2).T)
        toy_covariance = toy_covariance.reshape(
            (n_remaining_components, n_remaining_components, n_remaining_components, n_remaining_components)
        )
        profiled_information_covariance = toy_covariance / error_propagation_factor

        # Cross-check: toy mean
        toy_mean = np.mean(profiled_information_toys, axis=0)
        logger.debug("Central Fisher info:\n%s\nToy mean Fisher info:\n%s", profiled_information, toy_mean)

        return profiled_information, profiled_information_covariance

    # Calculate profiled information
    inverse_information_nuisance = np.linalg.inv(information_nuisance)
    profiled_information = information_phys - information_mix.T.dot(inverse_information_nuisance.dot(information_mix))

    return profiled_information
