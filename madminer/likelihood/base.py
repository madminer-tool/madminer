from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
from scipy.stats import poisson, norm

from ..analysis import DataAnalyzer
from ..utils.various import mdot

logger = logging.getLogger(__name__)


class BaseLikelihood(DataAnalyzer):
    def create_negative_log_likelihood(self, *args, **kwargs):
        raise NotImplementedError

    def create_expected_negative_log_likelihood(self, *args, **kwargs):
        raise NotImplementedError

    def _asimov_data(self, theta, test_split=0.2, sample_only_from_closest_benchmark=True, n_asimov=None):

        # get data
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

        # morphing
        theta_matrix = self._get_theta_benchmark_matrix(theta)
        weights_theta = mdot(theta_matrix, weights_benchmarks)
        weights_theta /= np.sum(weights_theta)

        return x, weights_theta

    def _log_likelihood(self, *args, **kwargs):
        raise NotImplementedError

    def _log_likelihood_kinematic(self, *args, **kwargs):
        raise NotImplementedError

    def _log_likelihood_poisson(
        self, n_observed, theta, nu, luminosity=300000.0, weights_benchmarks=None, total_weights=None
    ):

        if total_weights is not None:
            theta_matrix = self._get_theta_benchmark_matrix(theta)
            xsec = mdot(theta_matrix, total_weights)
        if weights_benchmarks is None:
            xsec = self.xsecs(thetas=[theta], nus=[nu], partition="train", generated_close_to=theta)[0][0]
        else:
            weights = self._weights([theta], [nu], weights_benchmarks)[0]
            xsec = sum(weights)

        n_predicted = xsec * luminosity
        if xsec < 0:
            logger.warning("Total cross section is negative (%s pb) at theta=%s)", xsec, theta)
            n_predicted = 10 ** -5
        n_observed_rounded = int(np.round(n_observed, 0))

        log_likelihood = poisson.logpmf(k=n_observed_rounded, mu=n_predicted)
        logger.debug(
            "Poisson log likelihood: %s (%s expected, %s observed at theta=%s)",
            log_likelihood,
            n_predicted,
            n_observed_rounded,
            theta,
        )
        return log_likelihood

    def _log_likelihood_constraint(self, nu):
        log_p = np.sum(norm.logpdf(nu))
        logger.debug("Constraint log likelihood: %s", log_p)
        return log_p
