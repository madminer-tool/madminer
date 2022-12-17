import logging

from abc import abstractmethod

import numpy as np

from scipy.stats import poisson
from scipy.stats import norm

from ..analysis import DataAnalyzer
from ..utils.various import mdot

logger = logging.getLogger(__name__)


class BaseLikelihood(DataAnalyzer):
    @abstractmethod
    def create_negative_log_likelihood(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def create_expected_negative_log_likelihood(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def _log_likelihood(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def _log_likelihood_kinematic(self, *args, **kwargs):
        raise NotImplementedError()

    def _asimov_data(
        self,
        theta,
        test_split=0.2,
        sample_only_from_closest_benchmark=True,
        n_asimov=None,
    ):

        # get data
        start_event, end_event, correction_factor = self._calculate_partition_bounds("test", test_split)
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

    def _log_likelihood_poisson(
        self,
        n_observed,
        theta,
        nu,
        luminosity=300000.0,
        weights_benchmarks=None,
        total_weights=None,
    ):
        if total_weights is not None and nu is None:
            # `histo` mode: Efficient morphing of whole cross section for the case without nuisance parameters
            theta_matrix = self._get_theta_benchmark_matrix(theta)
            xsec = mdot(theta_matrix, total_weights)
        elif total_weights is not None and self.nuisance_morpher is not None:
            # `histo` mode: Efficient morphing of whole cross section for the case with nuisance parameters
            logger.debug("Using nuisance interpolation")
            theta_matrix = self._get_theta_benchmark_matrix(theta)
            xsec = mdot(theta_matrix, total_weights)
            nuisance_effects = self.nuisance_morpher.calculate_nuisance_factors(
                nu, total_weights.reshape((1, -1))
            ).flatten()
            xsec *= nuisance_effects
        elif weights_benchmarks is not None:
            # `weighted` mode: Reweights existing events to (theta, nu) -- better than entirely new xsec calculation
            weights = self._weights([theta], [nu], weights_benchmarks)[0]
            xsec = sum(weights)
        elif weights_benchmarks is None:
            # `sampled` mode: Calculated total cross sections entirely new -- least efficient
            xsec = self.xsecs(thetas=[theta], nus=[nu], partition="train", generated_close_to=theta)[0][0]

        n_predicted = xsec * luminosity
        if xsec < 0:
            logger.warning("Total cross section is negative (%s pb) at theta=%s)", xsec, theta)
            n_predicted = 10**-5

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

    @staticmethod
    def _clean_nans(array):
        not_finite = np.any(~np.isfinite(array), axis=0)
        if np.sum(not_finite) > 0:
            logger.warning("Removing %s inf / nan results from calculation")
            array[:, not_finite] = 0.0
        return array
