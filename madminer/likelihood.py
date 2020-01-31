from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
from scipy.stats import poisson, norm, chi2

from madminer.analysis import DataAnalyzer
from madminer.utils.various import mdot, less_logging
from madminer.ml import ParameterizedRatioEstimator, Ensemble, LikelihoodEstimator, load_estimator

logger = logging.getLogger(__name__)


class CombinedLikelihood(DataAnalyzer):
    def create_negative_log_likelihood(
        self, model_file, x_observed, n_observed=None, x_observed_weights=None, include_xsec=True, luminosity=300000.0
    ):
        estimator = load_estimator(model_file)

        if n_observed is None:
            n_observed = len(x_observed)

        def nll(params):
            #Just return the expected Length
            if params is None:
                return self.n_nuisance_parameters+self.n_parameters
        
            #Process input
            if (len(params)!= self.n_nuisance_parameters+ self.n_parameters):
                logger.warning("Number of parameters is %s, expected %s physical parameters and %s nuisance paramaters",
                    len(params),self.n_parameters,self.n_nuisance_parameters )
            theta = params[: self.n_parameters]
            nu = params[self.n_parameters :]
            if len(nu)==0: nu=None

            #Compute Log Likelihood
            log_likelihood = self._log_likelihood(
                estimator, n_observed, x_observed, theta, nu, include_xsec, luminosity, x_observed_weights
            )
            return -log_likelihood

        return nll

    def create_expected_negative_log_likelihood(
        self, model_file, theta_true, nu_true, include_xsec=True, luminosity=300000.0, n_asimov=None
    ):
        x_asimov, x_weights = self._asimov_data(theta_true, n_asimov=n_asimov)
        n_observed = luminosity * self.xsecs([theta_true], [nu_true])[0]

        return self.create_negative_log_likelihood(
            model_file, x_asimov, n_observed, x_weights, include_xsec, luminosity
        )

    def fix_theta(self, nll, theta):
        def constrained_nll(params):
            params = np.concatenate((theta, params), axis=0)
            return nll(params)

        return constrained_nll

    def _asimov_data(self, theta, test_split=0.2, sample_only_from_closest_benchmark=True, n_asimov=None):
        start_event, end_event, correction_factor = self._train_test_split(False, test_split)
        x, weights_benchmarks = next(
            self.event_loader(
                start=start_event,
                end=end_event,
                batch_size=n_asimov,
                generated_close_to=theta if sample_only_from_closest_benchmark else None,
            )
        )
        weights_benchmarks *= correction_factor

        theta_matrix = self._get_theta_benchmark_matrix(theta)
        weights_theta = mdot(theta_matrix, weights_benchmarks)
        weights_theta /= np.sum(weights_theta)

        return x, weights_theta

    def _log_likelihood(
        self, estimator, n_events, xs, theta, nu, include_xsec=True, luminosity=300000.0, x_weights=None
    ):
        log_likelihood = 0.0
        if include_xsec:
            log_likelihood = log_likelihood + self._log_likelihood_poisson(n_events, theta, nu, luminosity)

        if x_weights is None:
            x_weights = n_events / float(len(xs)) * np.ones(len(xs))
        else:
            x_weights = x_weights * n_events / np.sum(x_weights)
        log_likelihood_events = self._log_likelihood_kinematic(estimator, xs, theta, nu)
        log_likelihood = log_likelihood + np.dot(x_weights, log_likelihood_events)

        if nu is not None:
            log_likelihood = log_likelihood + self._log_likelihood_constraint(nu)

        logger.debug("Total log likelihood: %s", log_likelihood)

        return log_likelihood

    def _log_likelihood_poisson(self, n_observed, theta, nu, luminosity=300000.0):
        xsec = self.xsecs(thetas=[theta], nus=[nu], partition="train", generated_close_to=theta)[0]
        n_predicted = xsec * luminosity
        n_observed_rounded = int(np.round(n_observed, 0))

        log_likelihood = poisson.logpmf(k=n_observed_rounded, mu=n_predicted)
        logger.debug(
            "Poisson log likelihood: %s (%s expected, %s observed)", log_likelihood, n_predicted, n_observed_rounded
        )
        return log_likelihood

    def _log_likelihood_kinematic(self, estimator, xs, theta, nu):
        if nu is not None:
            theta = np.concatenate((theta, nu), axis=0)

        if isinstance(estimator, ParameterizedRatioEstimator):
            with less_logging():
                log_r, _ = estimator.evaluate_log_likelihood_ratio(
                    x=xs, theta=theta.reshape((1, -1)), test_all_combinations=True, evaluate_score=False
                )
        elif isinstance(estimator, LikelihoodEstimator):
            with less_logging():
                log_r, _ = estimator.evaluate_log_likelihood(
                    x=xs, theta=theta.reshape((1, -1)), test_all_combinations=True, evaluate_score=False
                )
        elif isinstance(estimator, Ensemble) and estimator.estimator_type == "parameterized_ratio":
            with less_logging():
                log_r, _ = estimator.evaluate_log_likelihood_ratio(
                    x=xs,
                    theta=theta.reshape((1, -1)),
                    test_all_combinations=True,
                    evaluate_score=False,
                    calculate_covariance=False,
                )
        elif isinstance(estimator, Ensemble) and estimator.estimator_type == "likelihood":
            with less_logging():
                log_r, _ = estimator.evaluate_log_likelihood(
                    x=xs,
                    theta=theta.reshape((1, -1)),
                    test_all_combinations=True,
                    evaluate_score=False,
                    calculate_covariance=False,
                )
        else:
            raise NotImplementedError(
                "Likelihood (ratio) estimation is currently only implemented for "
                "ParameterizedRatioEstimator and LikelihoodEstimator and Ensemble instancees"
            )

        logger.debug("Kinematic log likelihood (ratio): %s", log_r.flatten())
        return log_r.flatten()

    def _log_likelihood_constraint(self, nu):
        log_p = np.sum(norm.logpdf(nu))
        logger.debug("Constraint log likelihood: %s", log_p)
        return log_p



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
    while the remaining N-M paramters are set to zero.
        
    Parameters
    ----------
    negative_log_likelihood : likelihood
        Function returned by Likelihood class (for example
        NeuralLikelihood.create_expected_negative_log_likelihood()`).
        
    remaining_components : list of int or None
        List with M entries, each an int with 0 <= remaining_compoinents[i] < N.
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
        
    thetas_eval : ndarray or None
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

    #DOF
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
        if len(grid_resolutions)!=m_paramaters:
            raise ValueError("Dimension of grid should be the same as number of remaining components!")
        theta_each = []
        for resolution, (theta_min, theta_max) in zip(grid_resolutions, grid_ranges):
            theta_each.append(np.linspace(theta_min, theta_max, resolution))
        theta_grid_each = np.meshgrid(*theta_each, indexing="ij")
        theta_grid_each = [theta.flatten() for theta in theta_grid_each]
        theta_grid_mdim = np.vstack(theta_grid_each).T
    else:
        theta_grid_mdim = thetas_eval
    
    #Obtain a theta_grid in n dimensions
    theta_grid_ndim=[]
    for theta_mdim in theta_grid_mdim:
        theta_ndim = np.zeros([n_parameters])
        for i,theta in zip(remaining_components,theta_mdim):
            theta_ndim[i] = theta
        theta_grid_ndim.append(theta_ndim)
    
    #evaluate -2 E[log r]
    log_r = np.array([-1.*negative_log_likelihood(theta) for theta in theta_grid_ndim])
    i_ml = np.argmax(log_r)
    log_r = log_r[:] - log_r[i_ml]
    m2_log_r = -2.*log_r
    p_value = chi2.sf(x=m2_log_r, df=dof)

    return theta_grid_mdim, p_value, i_ml, log_r



