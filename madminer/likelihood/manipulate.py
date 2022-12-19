import logging
import time

import numpy as np

from scipy.stats import chi2
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def fix_params(negative_log_likelihood, theta, fixed_components=None):
    """
    Function that reduces the dimensionality of a likelihood function by
    fixing some of the components.

    Parameters
    ----------
    negative_log_likelihood : likelihood
        Function returned by Likelihood class (for example
        NeuralLikelihood.create_expected_negative_log_likelihood()`)
        which takes an n-dimensional input parameter.

    theta : list of float
        m-dimensional vector of coordinate which will be fixed.

    fixed_components : list of int or None, optional.
        m-dimensional vector of coordinate indices provided in theta.
        `fixed_components=[0,1]` will fix the 1st and 2nd
        component of the parameter point. If None, uses [0, ..., m-1].

    Returns
    -------
    constrained_nll_negative_log_likelihood : likelihood
        Constrained likelihood function which takes an
        n-m dimensional input parameter.

    """
    if fixed_components is None:
        fixed_components = list(range(len(theta)))

    def constrained_nll(params):
        # Just return the expected Length
        n_dimension = negative_log_likelihood(None)
        if params is None:
            return n_dimension - len(fixed_components)

        # Process input
        if len(theta) != len(fixed_components):
            raise ValueError("Length of fixed_components and theta should be the same")
        if len(params) + len(fixed_components) != n_dimension:
            raise ValueError(f"Length of params should be {n_dimension-len(fixed_components)}")

        # Initialize full parameters
        params_full = np.zeros(n_dimension)

        # fill fixed components
        for icomp, thetacomp in zip(fixed_components, theta):
            params_full[icomp] = thetacomp

        # fill other components
        iparam = 0
        for i in range(len(params_full)):
            if i not in fixed_components:
                params_full[i] = params[iparam]
                iparam += 1

        # Return
        params_full = np.array(params_full)
        return negative_log_likelihood(params_full)

    return constrained_nll


def project_log_likelihood(
    negative_log_likelihood,
    remaining_components=None,
    grid_ranges=None,
    grid_resolutions=25,
    dof=None,
    thetas_eval=None,
):
    """
    Takes a likelihood function depending on N parameters, and evaluates
    for a set of M-dimensional parameter points (either grid or explicitly specified)
    while the remaining N-M parameters are set to zero.

    Parameters
    ----------
    negative_log_likelihood : likelihood
        Function returned by Likelihood class (for example
        NeuralLikelihood.create_expected_negative_log_likelihood()`).

    remaining_components : list of int or None , optional
        List with M entries, each an int with 0 <= remaining_components[i] < N.
        Denotes which parameters are kept, and their new order.
        All other parameters or projected out (set to zero). If None, all components
        are kept. Default: None

    grid_ranges : list of (tuple of float) or None, optional
        Specifies the boundaries of the parameter grid on which the p-values
        are evaluated. It should be `[(min, max), (min, max), ..., (min, max)]`,
        where the list goes over all parameters and `min` and `max` are
        float. If None, thetas_eval has to be given. Default: None.

    grid_resolutions : int or list of int, optional
        Resolution of the parameter space grid on which the p-values are
        evaluated. If int, the resolution is the same along every dimension
        of the hypercube. If list of int, the individual entries specify the number of
        points along each parameter individually. Doesn't have any effect if
        grid_ranges is None. Default value: 25.

    dof : int or None, optional
        If not None, sets the number of parameters for the calculation of the p-values.
        If None, the overall number of parameters is used. Default value: None.

    thetas_eval : ndarray or None , optional
        Manually specifies the parameter point at which the likelihood and p-values
        are evaluated. If None, grid_ranges and resolution are used instead to construct
        a regular grid. Default value: None.

    Returns
    -------
    parameter_grid : ndarray
        Parameter points at which the p-values are evaluated with shape
        `(n_grid_points, n_parameters)`.

    p_values : ndarray
        Observed p-values for each parameter point on the grid,
        with shape `(n_grid_points,)`.

    mle : int
        Index of the parameter point with the best fit (largest p-value
        / smallest -2 log likelihood ratio).

    log_likelihood_ratio : ndarray or None
        log likelihood ratio based only on kinematics for each point of the grid,
        with shape `(n_grid_points,)`.

    """

    # Components
    n_parameters = negative_log_likelihood(None)
    if remaining_components is None:
        remaining_components = range(n_parameters)
    m_paramaters = len(remaining_components)

    # DOF
    if dof is None:
        dof = m_paramaters

    # Theta grid
    if thetas_eval is None and grid_resolutions is None:
        raise ValueError("thetas_eval and grid_resolutions cannot both be None")
    elif thetas_eval is not None and grid_resolutions is not None:
        raise ValueError("thetas_eval and grid_resolutions cannot both be set, make up your mind!")
    elif thetas_eval is None:
        if isinstance(grid_resolutions, int):
            grid_resolutions = [grid_resolutions for _ in range(grid_ranges)]
        if len(grid_resolutions) != m_paramaters:
            raise ValueError("Dimension of grid should be the same as number of remaining components!")
        theta_each = []
        for resolution, (theta_min, theta_max) in zip(grid_resolutions, grid_ranges):
            theta_each.append(np.linspace(theta_min, theta_max, resolution))
        theta_grid_each = np.meshgrid(*theta_each, indexing="ij")
        theta_grid_each = [theta.flatten() for theta in theta_grid_each]
        theta_grid_mdim = np.vstack(theta_grid_each).T
    else:
        theta_grid_mdim = thetas_eval

    # Obtain a theta_grid in n dimensions
    theta_grid_ndim = []
    for theta_mdim in theta_grid_mdim:
        theta_ndim = np.zeros([n_parameters])
        for i, theta in zip(remaining_components, theta_mdim):
            theta_ndim[i] = theta
        theta_grid_ndim.append(theta_ndim)

    # evaluate -2 E[log r]
    log_r = np.array([-1.0 * negative_log_likelihood(theta) for theta in theta_grid_ndim])
    i_ml = np.argmax(log_r)
    log_r = log_r[:] - log_r[i_ml]
    m2_log_r = -2.0 * log_r
    p_value = chi2.sf(x=m2_log_r, df=dof)

    return theta_grid_mdim, p_value, i_ml, log_r


def profile_log_likelihood(
    negative_log_likelihood,
    remaining_components=None,
    grid_ranges=None,
    grid_resolutions=25,
    thetas_eval=None,
    theta_start=None,
    dof=None,
    method="TNC",
):
    """
    Takes a likelihood function depending on N parameters, and evaluates
    for a set of M-dimensional parameter points (either grid or explicitly specified)
    while the remaining N-M parameters are profiled over.

    Parameters
    ----------
    negative_log_likelihood : likelihood
        Function returned by Likelihood class (for example
        NeuralLikelihood.create_expected_negative_log_likelihood()`).

    remaining_components : list of int or None , optional
        List with M entries, each an int with 0 <= remaining_components[i] < N.
        Denotes which parameters are kept, and their new order.
        All other parameters or projected out (set to zero). If None, all components
        are kept. Default: None

    grid_ranges : list of (tuple of float) or None, optional
        Specifies the boundaries of the parameter grid on which the p-values
        are evaluated. It should be `[(min, max), (min, max), ..., (min, max)]`,
        where the list goes over all parameters and `min` and `max` are
        float. If None, thetas_eval has to be given. Default: None.

    grid_resolutions : int or list of int, optional
        Resolution of the parameter space grid on which the p-values are
        evaluated. If int, the resolution is the same along every dimension
        of the hypercube. If list of int, the individual entries specify the number of
        points along each parameter individually. Doesn't have any effect if
        grid_ranges is None. Default value: 25.

    thetas_eval : ndarray or None , optional
        Manually specifies the parameter point at which the likelihood and p-values
        are evaluated. If None, grid_ranges and resolution are used instead to construct
        a regular grid. Default value: None.

    theta_start : ndarray or None , optional
        Manually specifies a parameter point which is the starting point
        for the minimization algorithm which find the maximum likelihood point.
        If None, theta_start = 0 is used.
        Default is None.

    dof : int or None, optional
        If not None, sets the number of parameters for the calculation of the p-values.
        If None, the overall number of parameters is used. Default value: None.

    method : {"TNC", " L-BFGS-B"} , optional
        Minimization method used. Default value: "TNC"

    Returns
    -------
    parameter_grid : ndarray
        Parameter points at which the p-values are evaluated with shape
        `(n_grid_points, n_parameters)`.

    p_values : ndarray
        Observed p-values for each parameter point on the grid,
        with shape `(n_grid_points,)`.

    mle : int
        Index of the parameter point with the best fit (largest p-value
        / smallest -2 log likelihood ratio).

    log_likelihood_ratio : ndarray or None
        log likelihood ratio based only on kinematics for each point of the grid,
        with shape `(n_grid_points,)`.

    """

    # Components
    n_parameters = negative_log_likelihood(None)
    if remaining_components is None:
        remaining_components = range(n_parameters)
    m_parameters = len(remaining_components)

    # DOF
    if dof is None:
        dof = m_parameters

    # Method
    if method not in {"TNC", " L-BFGS-B"}:
        raise ValueError(f"Method {method} unknown.")

    # Initial guess for theta
    if theta_start is None:
        theta_start = np.zeros(n_parameters)

    # Theta grid
    if thetas_eval is None and grid_resolutions is None:
        raise ValueError("thetas_eval and grid_resolutions cannot both be None")
    elif thetas_eval is not None and grid_resolutions is not None:
        raise ValueError("thetas_eval and grid_resolutions cannot both be set, make up your mind!")
    elif thetas_eval is None:
        if isinstance(grid_resolutions, int):
            grid_resolutions = [grid_resolutions for _ in range(grid_ranges)]
        if len(grid_resolutions) != m_parameters:
            raise ValueError("Dimension of grid should be the same as number of remaining components!")
        theta_each = []
        for resolution, (theta_min, theta_max) in zip(grid_resolutions, grid_ranges):
            theta_each.append(np.linspace(theta_min, theta_max, resolution))
        theta_grid_each = np.meshgrid(*theta_each, indexing="ij")
        theta_grid_each = [theta.flatten() for theta in theta_grid_each]
        theta_grid_mdim = np.vstack(theta_grid_each).T
    else:
        theta_grid_mdim = thetas_eval

    # Obtain global minimum - Eq.(59) in 1805.00020
    result = minimize(negative_log_likelihood, x0=theta_start, method=method)
    best_fit_global = result.x

    # scan over grid
    log_r = []
    pscan = 0.01
    start_time = time.time()
    for iscan, theta_mdim in enumerate(theta_grid_mdim):
        # logger output
        if iscan / len(theta_grid_mdim) > pscan:
            elapsed_time = time.time() - start_time
            logger.info("Processed %s %% of parameter points in %.1f seconds.", pscan * 100, elapsed_time)
            while iscan / len(theta_grid_mdim) > pscan:
                if pscan > 0.095:
                    pscan += 0.1
                else:
                    pscan += 0.01

        # fix some parameters
        constrained_negative_log_likelihood = fix_params(
            negative_log_likelihood, theta=theta_mdim, fixed_components=remaining_components
        )

        # obtain starting point
        theta0 = []
        for i, theta in enumerate(theta_start):
            if i not in remaining_components:
                theta0.append(theta)

        # obtain local minimum - Eq.(58) in 1805.00020
        result = minimize(constrained_negative_log_likelihood, x0=np.array(theta0), method=method)
        best_fit_constrained = result.x

        # Expected Log Likelihood - Eq.(57) in 1805.00020
        profiled_logr = -1.0 * (
            constrained_negative_log_likelihood(best_fit_constrained) - negative_log_likelihood(best_fit_global)
        )
        log_r.append(profiled_logr)

    # evaluate p_value and best fit point
    logr = np.array(log_r)
    i_ml = np.argmax(log_r)
    log_r = log_r[:] - log_r[i_ml]
    m2_log_r = -2.0 * log_r
    p_value = chi2.sf(x=m2_log_r, df=dof)

    return theta_grid_mdim, p_value, i_ml, log_r
