from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np

from ..utils.various import less_logging
from ..ml import ParameterizedRatioEstimator, Ensemble, LikelihoodEstimator, load_estimator
from .base import BaseLikelihood

logger = logging.getLogger(__name__)


class NeuralLikelihood(BaseLikelihood):
    def create_negative_log_likelihood(
        self,
        model_file,
        x_observed,
        n_observed=None,
        x_observed_weights=None,
        include_xsec=True,
        luminosity=300000.0,
        mode="weighted",
        n_weighted=10000,
    ):
        estimator = load_estimator(model_file)

        if n_observed is None:
            n_observed = len(x_observed)

        # Weighted sampled
        if mode == "weighted":
            weights_benchmarks = self._get_weights_benchmarks(n_toys=n_weighted, test_split=None)
        else:
            weights_benchmarks = None

        def nll(params):
            # Just return the expected Length
            if params is None:
                return self.n_nuisance_parameters + self.n_parameters

            # Process input
            if len(params) != self.n_nuisance_parameters + self.n_parameters:
                logger.warning(
                    "Number of parameters is %s, expected %s physical parameters and %s nuisance paramaters",
                    len(params),
                    self.n_parameters,
                    self.n_nuisance_parameters,
                )
            theta = params[: self.n_parameters]
            nu = params[self.n_parameters :]
            if len(nu) == 0:
                nu = None

            # Compute Log Likelihood
            log_likelihood = self._log_likelihood(
                estimator,
                n_observed,
                x_observed,
                theta,
                nu,
                include_xsec,
                luminosity,
                x_observed_weights,
                weights_benchmarks,
            )
            return -log_likelihood

        return nll

    def create_expected_negative_log_likelihood(
        self,
        model_file,
        theta_true,
        nu_true,
        include_xsec=True,
        luminosity=300000.0,
        n_asimov=None,
        mode="sampled",
        n_weighted=10000,
    ):
        x_asimov, x_weights = self._asimov_data(theta_true, n_asimov=n_asimov)
        n_observed = luminosity * self.xsecs([theta_true], [nu_true])[0]

        return self.create_negative_log_likelihood(
            model_file, x_asimov, n_observed, x_weights, include_xsec, luminosity, mode, n_weighted
        )

    def _log_likelihood(
        self,
        estimator,
        n_events,
        xs,
        theta,
        nu,
        include_xsec=True,
        luminosity=300000.0,
        x_weights=None,
        weights_benchmarks=None,
    ):
        """
        Low-level function which calculates the value of the log-likelihood ratio.
        See create_negative_log_likelihood for options.   
        """

        log_likelihood = 0.0
        if include_xsec:
            log_likelihood = log_likelihood + self._log_likelihood_poisson(
                n_events, theta, nu, luminosity, weights_benchmarks
            )

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

    def _log_likelihood_kinematic(self, estimator, xs, theta, nu):
        """
        Low-level function which calculates the value of the kinematic part of the
        log-likelihood. See create_negative_log_likelihood for options.
        """

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
        log_r = log_r.flatten()
        log_r = log_r.astype(np.float64)
        log_r = self._clean_nans(log_r)
        return log_r

    def _get_weights_benchmarks(self, n_toys, test_split=None):
        """
        Low-level function that creates weighted events and returns weights
        """

        start_event, end_event, correction_factor = self._train_test_split(True, test_split)
        x, weights_benchmarks = self.weighted_events(start_event=start_event, end_event=end_event, n_draws=n_toys)
        weights_benchmarks *= self.n_samples / n_toys

        return weights_benchmarks

    @staticmethod
    def _clean_nans(array):
        not_finite = np.any(~np.isfinite(array), axis=0)
        if np.sum(not_finite) > 0:
            logger.warning("Removing %s inf / nan results from calculation")
            array[:, not_finite] = 0.0
        return array
