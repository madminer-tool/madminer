from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
import six
import os
from scipy.stats import chi2, poisson

from madminer.utils.interfaces.madminer_hdf5 import load_madminer_settings, madminer_event_loader
from madminer.utils.analysis import get_theta_benchmark_matrix, get_dtheta_benchmark_matrix, mdot
from madminer.utils.morphing import PhysicsMorpher, NuisanceMorpher
from madminer.utils.various import format_benchmark, math_commands, weighted_quantile, sanitize_array
from madminer.ml import ParameterizedRatioEstimator, Ensemble

logger = logging.getLogger(__name__)


class AsymptoticLimits:
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
        if include_nuisance_parameters:
            raise NotImplementedError("Nuisance parameters are not yet supported!")

        # Save settings
        self.madminer_filename = filename
        self.include_nuisance_parameters = include_nuisance_parameters

        logger.info("Loading data from %s", filename)

        # Load data
        (
            self.parameters,
            self.benchmarks,
            self.benchmark_is_nuisance,
            self.morphing_components,
            self.morphing_matrix,
            self.observables,
            self.n_samples,
            _,
            self.reference_benchmark,
            self.nuisance_parameters,
        ) = load_madminer_settings(filename, include_nuisance_benchmarks=include_nuisance_parameters)
        self.n_parameters = len(self.parameters)
        self.n_benchmarks = len(self.benchmarks)
        self.n_benchmarks_phys = np.sum(np.logical_not(self.benchmark_is_nuisance))

        self.n_nuisance_parameters = 0
        if self.nuisance_parameters is not None and include_nuisance_parameters:
            self.n_nuisance_parameters = len(self.nuisance_parameters)
        else:
            self.nuisance_parameters = None

        logger.info("Found %s parameters", len(self.parameters))
        for key, values in six.iteritems(self.parameters):
            logger.debug(
                "   %s (LHA: %s %s, maximal power in squared ME: %s, range: %s)",
                key,
                values[0],
                values[1],
                values[2],
                values[3],
            )

        if self.nuisance_parameters is not None and include_nuisance_parameters:
            logger.info("Found %s nuisance parameters", self.n_nuisance_parameters)
            for key, values in six.iteritems(self.nuisance_parameters):
                logger.debug("   %s (%s)", key, values)
        elif include_nuisance_parameters:
            self.include_nuisance_parameters = False
            logger.warning("Did not find nuisance parameters!")

        logger.info("Found %s benchmarks, of which %s physical", self.n_benchmarks, self.n_benchmarks_phys)
        for (key, values), is_nuisance in zip(six.iteritems(self.benchmarks), self.benchmark_is_nuisance):
            if is_nuisance:
                logger.debug("   %s: nuisance parameter", key)
            else:
                logger.debug("   %s: %s", key, format_benchmark(values))

        logger.info("Found %s observables: %s", len(self.observables), ", ".join(self.observables))
        logger.info("Found %s events", self.n_samples)

        # Morphing
        self.morpher = None
        if self.morphing_matrix is not None and self.morphing_components is not None:
            self.morpher = PhysicsMorpher(self.parameters)
            self.morpher.set_components(self.morphing_components)
            self.morpher.set_basis(self.benchmarks, morphing_matrix=self.morphing_matrix)

            logger.info("Found morphing setup with %s components", len(self.morphing_components))

        else:
            raise RuntimeError("Did not find morphing setup.")

        # Nuisance morphing
        self.nuisance_morpher = None
        if self.include_nuisance_parameters:
            self.nuisance_morpher = NuisanceMorpher(
                self.nuisance_parameters, list(self.benchmarks.keys()), self.reference_benchmark
            )
            logger.info("Found nuisance morphing setup")

    def observed_limits(self, x_observed, theta_ranges, model_file, resolution=25, luminosity=300000.0):
        theta_grid = self._make_theta_grid(theta_ranges, resolution)
        p_values, i_ml = self._analyse(x_observed, theta_grid, model_file, len(x_observed), luminosity)
        return theta_grid, p_values, i_ml

    def expected_limits(self, theta_true, theta_ranges, model_file, resolution=25, luminosity=300000.0):
        x_asimov, x_weights = self._asimov_data(theta_true)
        n_observed = luminosity * self._calculate_xsecs([theta_true])[0]
        theta_grid = self._make_theta_grid(theta_ranges, resolution)
        p_values, i_ml = self._analyse(x_asimov, theta_grid, model_file, n_observed, x_weights, luminosity)
        return theta_grid, p_values, i_ml

    def asymptotic_p_value(self, log_likelihood_ratio):
        q = -2.0 * log_likelihood_ratio
        p_value = chi2.sf(x=q, df=self.n_parameters)
        return p_value

    def _analyse(self, x, theta_grid, model_file, n_events, obs_weights=None, luminosity=300000.0):
        # Observation weights
        if obs_weights is None:
            obs_weights = np.ones(len(x))
        obs_weights /= np.sum(obs_weights)

        # Kinematic part
        model = self._load_model(model_file)
        log_r_kin = self._calculate_log_likelihood_ratio_kinematics(x, theta_grid, model)
        log_r_kin = n_events / len(x) * np.einsum("tx,x->t", log_r_kin, obs_weights)

        # xsec part
        log_p_xsec = self._calculate_log_likelihood_xsec(n_events, theta_grid, luminosity)

        # Combine and get p-values
        log_r = log_r_kin + log_p_xsec
        log_r, i_ml = self._subtract_ml(log_r)
        p_values = self.asymptotic_p_value(log_r)
        return p_values, i_ml

    def _load_model(self, filename):
        if os.path.isdir(filename):
            model = Ensemble()
            model.load(filename)
        else:
            model = ParameterizedRatioEstimator()
            model.load(filename)

        return model

    def _calculate_xsecs(self, thetas, test_split=0.2):
        # Test split
        start_event, end_event = self._train_test_split(self, False, test_split)

        # Total xsecs for benchmarks
        xsecs_benchmarks = 0.0
        for observations, weights in madminer_event_loader(self.madminer_filename, start=start_event, end=end_event):
            xsecs_benchmarks += np.sum(weights, axis=0)

        # xsecs at thetas
        xsecs = []
        for theta in thetas:
            theta_matrix = get_theta_benchmark_matrix("morphing", theta, self.benchmarks, self.morpher)
            xsecs.append(mdot(theta_matrix, xsecs_benchmarks))
        return np.asarray(xsecs)

    def _asimov_data(self, theta, test_split=0.2):
        start_event, end_event = self._train_test_split(False, test_split)
        x, weights_benchmarks = next(
            madminer_event_loader(self.madminer_filename, start=start_event, end=end_event, batch_size=None)
        )

        theta_matrix = get_theta_benchmark_matrix("morphing", theta, self.benchmarks, self.morpher)
        weights_theta = mdot(theta_matrix, weights_benchmarks)
        weights_theta /= np.sum(weights_theta)

        return x, weights_theta

    @staticmethod
    def _make_theta_grid(theta_ranges, resolution):
        theta_each = []
        for theta_min, theta_max in theta_ranges:
            theta_each.append(np.linspace(theta_min, theta_max, resolution))
        theta_grid_each = np.meshgrid(*theta_each)
        theta_grid_each = [theta.flatten() for theta in theta_grid_each]
        theta_grid = np.vstack(theta_grid_each).T
        return theta_grid

    def _calculate_log_likelihood_xsec(self, n_observed, theta_grid, luminosity=300000.0):
        n_predicted = self._calculate_xsec(theta_grid) * luminosity
        log_p = poisson.logpmf(k=n_observed, mu=n_predicted)
        return log_p

    def _calculate_log_likelihood_ratio_kinematics(self, x_observed, theta_grid, model, theta1=None):
        if isinstance(model, ParameterizedRatioEstimator):
            log_r, _ = model.evaluate_log_likelihood_ratio(
                x=x_observed, theta=theta_grid, evaluate_all_combinations=True, calculate_score=False
            )
        elif isinstance(model, Ensemble) and model.estimator_type == "parameterized_ratio":
            log_r, _ = model.evaluate_log_likelihood_ratio(
                x=x_observed,
                theta=theta_grid,
                evaluate_all_combinations=True,
                calculate_score=False,
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
    def _train_test_split(train, test_split):
        """
        Returns the start and end event for train samples (train = True) or test samples (train = False).

        Parameters
        ----------
        train : bool
            True if training data is generated, False if test data is generated.

        test_split : float
            Fraction of events reserved for testing.

        Returns
        -------
        start_event : int
            Index of the first unweighted event to consider.

        end_event : int
            Index of the last unweighted event to consider.

        """
        if train:
            start_event = 0

            if test_split is None or test_split <= 0.0 or test_split >= 1.0:
                end_event = None
            else:
                end_event = int(round((1.0 - test_split) * self.n_samples, 0))
                if end_event < 0 or end_event > self.n_samples:
                    raise ValueError("Irregular train / test split: sample {} / {}", end_event, self.n_samples)

        else:
            if test_split is None or test_split <= 0.0 or test_split >= 1.0:
                start_event = 0
            else:
                start_event = int(round((1.0 - test_split) * self.n_samples, 0)) + 1
                if start_event < 0 or start_event > self.n_samples:
                    raise ValueError("Irregular train / test split: sample {} / {}", start_event, self.n_samples)

            end_event = None

        return start_event, end_event
