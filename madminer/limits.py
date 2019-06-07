from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
import os
from scipy.stats import chi2, poisson

from madminer.analysis import DataAnalyzer
from madminer.utils.various import mdot
from madminer.ml import ParameterizedRatioEstimator, Ensemble, ScoreEstimator, load_estimator
from madminer.utils.histo import Histo
from madminer.sampling import SampleAugmenter
from madminer import sampling

logger = logging.getLogger(__name__)


class AsymptoticLimits(DataAnalyzer):
    """
    Functions to calculate observed and expected constraints, using asymptotic properties of the likelihood ratio as
    test statistics.

    Parameters
    ----------
    filename : str
        Path to MadMiner file (for instance the output of `madminer.delphes.DelphesProcessor.save()`).

    include_nuisance_parameters : bool, optional
        If True, nuisance parameters are taken into account. Default value: False.
    """

    def __init__(self, filename=None, include_nuisance_parameters=False):
        super(AsymptoticLimits, self).__init__(filename, False, include_nuisance_parameters)

    def observed_limits(
        self,
        x_observed,
        theta_ranges,
        mode="ml",
        model_file=None,
        hist_vars=None,
        hist_bins=20,
        include_xsec=True,
        resolutions=25,
        luminosity=300000.0,
        n_toys_per_theta=10000,
        returns="pval",
        dof=None,
        n_observed=None,
        histo_theta_batchsize=100,
    ):
        if n_observed is None:
            n_observed = len(x_observed)
        theta_grid, return_values, i_ml = self._analyse(
            n_observed,
            x_observed,
            theta_ranges,
            resolutions,
            mode,
            model_file,
            hist_vars,
            hist_bins,
            include_xsec,
            None,
            luminosity,
            n_toys_per_theta,
            returns=returns,
            dof=dof,
            histo_theta_batchsize=histo_theta_batchsize,
        )
        return theta_grid, return_values, i_ml

    def expected_limits(
        self,
        theta_true,
        theta_ranges,
        mode="ml",
        model_file=None,
        hist_vars=None,
        hist_bins=20,
        include_xsec=True,
        resolutions=25,
        luminosity=300000.0,
        n_toys_per_theta=10000,
        returns="pval",
        dof=None,
        histo_theta_batchsize=100,
    ):
        logger.info("Generating Asimov data")
        x_asimov, x_weights = self._asimov_data(theta_true)
        n_observed = luminosity * self._calculate_xsecs([theta_true])[0]
        logger.info("Expected events: %s", n_observed)
        theta_grid, return_values, i_ml = self._analyse(
            n_observed,
            x_asimov,
            theta_ranges,
            resolutions,
            mode,
            model_file,
            hist_vars,
            hist_bins,
            include_xsec,
            x_weights,
            luminosity,
            n_toys_per_theta,
            returns=returns,
            dof=dof,
            histo_theta_batchsize=histo_theta_batchsize,
        )
        return theta_grid, return_values, i_ml

    def asymptotic_p_value(self, log_likelihood_ratio, dof=None):
        if dof is None:
            dof = self.n_parameters
        q = -2.0 * log_likelihood_ratio
        p_value = chi2.sf(x=q, df=dof)
        return p_value

    def _analyse(
        self,
        n_events,
        x,
        theta_ranges,
        theta_resolutions,
        mode="ml",
        model_file=None,
        hist_vars=None,
        hist_bins=20,
        include_xsec=True,
        obs_weights=None,
        luminosity=300000.0,
        n_toys_per_theta=10000,
        returns="pval",
        dof=None,
        histo_theta_batchsize=100,
    ):
        logger.debug("Calculating p-values for %s expected events", n_events)

        assert returns in ["pval", "llr", "llr_raw"], "returns has to be either 'pval','llr' or 'llr_raw'!"

        # Observation weights
        if obs_weights is None:
            obs_weights = np.ones(len(x))
        obs_weights /= np.sum(obs_weights)
        obs_weights = obs_weights.astype(np.float64)

        # Theta grid
        theta_grid = self._make_theta_grid(theta_ranges, theta_resolutions)

        # Kinematic part
        if mode == "rate":
            log_r_kin = 0.0
        elif mode == "ml":
            assert model_file is not None
            logger.info("Loading kinematic likelihood ratio estimator")
            model = load_estimator(model_file)

            logger.info("Calculating kinematic log likelihood ratio with estimator")
            log_r_kin = self._calculate_log_likelihood_ratio_kinematics(x, theta_grid, model)
            log_r_kin = log_r_kin.astype(np.float64)
            log_r_kin = self._clean_nans(log_r_kin)
            logger.debug("Raw mean -2 log r: %s", np.mean(-2.0 * log_r_kin, axis=1))
            log_r_kin = n_events * np.sum(log_r_kin * obs_weights[np.newaxis, :], axis=1)
            logger.debug("Rescaled -2 log r: %s", -2.0 * log_r_kin)

        elif mode == "histo":
            if hist_vars is not None:
                logger.info("Setting up standard summary statistics")
                summary_function = self._make_summary_statistic_function("observables", observables=hist_vars)
            elif model_file is not None:
                logger.info("Loading score estimator and setting it up as summary statistics")
                model = load_estimator(model_file)
                summary_function = self._make_summary_statistic_function("sally", model=model)
            else:
                raise RuntimeError("For 'histo' mode, either provide histo_vars or model_file!")
            summary_stats = summary_function(x)
            del x

            logger.info("Creating histogram with %s bins for the summary statistics", hist_bins)
            histo = self._make_histo(
                summary_function,
                hist_bins,
                theta_grid,
                theta_resolutions,
                n_toys_per_theta,
                histo_theta_batchsize=histo_theta_batchsize,
            )

            logger.info("Calculating kinematic log likelihood with histograms")
            log_r_kin = self._calculate_log_likelihood_histo(summary_stats, theta_grid, histo)
            log_r_kin = log_r_kin.astype(np.float64)
            log_r_kin = self._clean_nans(log_r_kin)
            log_r_kin = n_events * np.sum(log_r_kin * obs_weights[np.newaxis, :], axis=1)

        else:
            raise ValueError("Unknown mode {}, has to be 'ml' or 'histo' or 'xsec'".format(mode))

        # xsec part
        if include_xsec:
            logger.info("Calculating rate log likelihood")
            log_p_xsec = self._calculate_log_likelihood_xsec(n_events, theta_grid, luminosity)
            logger.debug("Rate -2 log p: %s", -2.0 * log_p_xsec)
        else:
            log_p_xsec = 0.0

        # Combine and get p-values
        logger.info("Calculating p-values")
        log_r = log_r_kin + log_p_xsec
        if returns == "llr_raw":
            return theta_grid, log_r, 0

        logger.debug("Combined -2 log r: %s", -2.0 * log_r)
        log_r, i_ml = self._subtract_ml(log_r)
        logger.debug("Min-subtracted -2 log r: %s", -2.0 * log_r)
        p_values = self.asymptotic_p_value(log_r, dof=dof)

        if returns == "llr":
            return theta_grid, log_r, i_ml
        return theta_grid, p_values, i_ml

    def _make_summary_statistic_function(self, mode, model=None, observables=None):
        if mode == "observables":
            assert observables is not None
            x_indices = self._find_x_indices(observables)

            logger.debug("Preparing observables %s as summary statistic function", x_indices)

            def summary_function(x):
                return x[:, x_indices]

        elif mode == "sally":
            assert isinstance(model, ScoreEstimator)

            logger.debug("Preparing score estimator as summary statistic function")

            def summary_function(x):
                score = model.evaluate_score(x)
                score = score[:, : self.n_parameters]
                return score

        else:
            raise RuntimeError("Unknown mode {}, has to be 'observables' or 'sally'".format(mode))

        return summary_function

    def _calculate_xsecs(self, thetas, test_split=0.2):
        # Test split
        start_event, end_event, correction_factor = self._train_test_split(False, test_split)

        # Total xsecs for benchmarks
        xsecs_benchmarks = 0.0
        for observations, weights in self.event_loader(start=start_event, end=end_event):
            xsecs_benchmarks += np.sum(weights, axis=0)

        # xsecs at thetas
        xsecs = []
        for theta in thetas:
            theta_matrix = self._get_theta_benchmark_matrix(theta)
            xsecs.append(mdot(theta_matrix, xsecs_benchmarks) * correction_factor)
        return np.asarray(xsecs)

    def _asimov_data(self, theta, test_split=0.2):
        start_event, end_event, correction_factor = self._train_test_split(False, test_split)
        x, weights_benchmarks = next(self.event_loader(start=start_event, end=end_event, batch_size=None))
        weights_benchmarks *= correction_factor

        theta_matrix = self._get_theta_benchmark_matrix(theta)
        weights_theta = mdot(theta_matrix, weights_benchmarks)
        weights_theta /= np.sum(weights_theta)

        return x, weights_theta

    @staticmethod
    def _make_theta_grid(theta_ranges, resolutions):
        if isinstance(resolutions, int):
            resolutions = [resolutions for _ in range(theta_ranges)]
        theta_each = []
        for resolution, (theta_min, theta_max) in zip(resolutions, theta_ranges):
            theta_each.append(np.linspace(theta_min, theta_max, resolution))
        theta_grid_each = np.meshgrid(*theta_each, indexing="ij")
        theta_grid_each = [theta.flatten() for theta in theta_grid_each]
        theta_grid = np.vstack(theta_grid_each).T
        return theta_grid

    def _make_histo(
        self, summary_function, x_bins, theta_grid, theta_bins, n_toys_per_theta=1000, histo_theta_batchsize=100
    ):
        logger.info("Building histogram with %s bins per parameter and %s bins per observable", theta_bins, x_bins)
        histo = Histo(theta_bins, x_bins)
        logger.debug("Generating histo data")
        theta, summary_stats = self._make_histo_data(
            summary_function, theta_grid, n_toys_per_theta, histo_theta_batchsize=histo_theta_batchsize
        )
        logger.debug(
            "Histo data has theta dimensions %s and summary stats dimensions %s", theta.shape, summary_stats.shape
        )
        logger.debug("Filling histogram with summary statistics")
        histo.fit(theta, summary_stats, fill_empty_bins=True)
        return histo

    def _make_histo_data(self, summary_function, thetas, n_toys_per_theta, test_split=0.2, histo_theta_batchsize=100):
        sampler = SampleAugmenter(self.madminer_filename, include_nuisance_parameters=self.include_nuisance_parameters)
        all_summary_stats, all_theta = None, None

        n_thetas = len(thetas)
        n_batches = (n_thetas - 1) // histo_theta_batchsize + 1
        for i_batch in range(n_batches):
            logger.debug("Generating histogram data for batch %s / %s", i_batch + 1, n_batches)
            theta_batch = thetas[i_batch * histo_theta_batchsize : (i_batch + 1) * histo_theta_batchsize]
            logger.debug(
                "Theta data: indices %s to %s, shape %s",
                i_batch * histo_theta_batchsize,
                (i_batch + 1) * histo_theta_batchsize,
                theta_batch.shape,
            )
            x, theta, _ = sampler.sample_train_plain(
                theta=sampling.morphing_points(theta_batch),
                n_samples=n_toys_per_theta * len(theta_batch),
                test_split=test_split,
                filename=None,
                folder=None,
                suppress_logging=True,
            )
            summary_stats = summary_function(x)
            logger.debug(
                "Output: x has shape %s, summary_stats %s, theta %s", x.shape, summary_stats.shape, theta.shape
            )
            if all_theta is None or all_summary_stats is None:
                all_theta = theta
                all_summary_stats = summary_stats
            else:
                all_theta = np.concatenate((all_theta, theta), 0)
                all_summary_stats = np.concatenate((all_summary_stats, summary_stats), 0)
        return all_theta, all_summary_stats

    def _find_x_indices(self, observables):
        x_names = list(self.observables.keys())
        x_indices = []
        for obs in observables:
            try:
                x_indices.append(x_names.index(obs))
            except ValueError:
                raise RuntimeError("Unknown observable {}, has to be one of {}".format(obs, x_names))
        logger.debug("Using x indices %s", x_indices)
        return x_indices

    @staticmethod
    def _calculate_log_likelihood_histo(x, theta_grid, histo):
        log_p = []
        for theta in theta_grid:
            log_p.append(histo.log_likelihood(theta, x))
        log_p = np.asarray(log_p)
        return log_p

    def _calculate_log_likelihood_xsec(self, n_observed, theta_grid, luminosity=300000.0):
        n_observed_rounded = int(np.round(n_observed, 0))
        n_predicted = self._calculate_xsecs(theta_grid) * luminosity
        logger.debug("Observed events: %s", n_observed)
        logger.debug("Expected events: %s", n_predicted)
        log_p = poisson.logpmf(k=n_observed_rounded, mu=n_predicted)
        return log_p

    def _calculate_log_likelihood_ratio_kinematics(self, x_observed, theta_grid, model, theta1=None):
        if isinstance(model, ParameterizedRatioEstimator):
            log_r, _ = model.evaluate_log_likelihood_ratio(
                x=x_observed, theta=theta_grid, test_all_combinations=True, evaluate_score=False
            )
        elif isinstance(model, Ensemble) and model.estimator_type == "parameterized_ratio":
            log_r, _ = model.evaluate_log_likelihood_ratio(
                x=x_observed,
                theta=theta_grid,
                test_all_combinations=True,
                evaluate_score=False,
                calculate_covariance=False,
            )
        else:
            raise NotImplementedError(
                "Likelihood ratio estimation is currently only implemented for "
                "ParameterizedRatioEstimator instancees"
            )
        return log_r

    @staticmethod
    def _subtract_ml(log_r):
        i_ml = np.argmax(log_r)
        log_r_subtracted = log_r[:] - log_r[i_ml]
        return log_r_subtracted, i_ml

    @staticmethod
    def _clean_nans(array):
        not_finite = np.any(~np.isfinite(array), axis=0)
        if np.sum(not_finite) > 0:
            logger.warning("Removing %s inf / nan results from calculation")
            array[:, not_finite] = 0.0
        return array
