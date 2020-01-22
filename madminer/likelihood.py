from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
from scipy.stats import poisson, norm

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
            theta = params[: self.n_parameters]
            nu = params[self.n_parameters :]
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
