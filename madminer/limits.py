from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
from scipy.stats import chi2, poisson

from madminer.analysis import DataAnalyzer
from madminer.utils.various import mdot
from madminer.ml import ParameterizedRatioEstimator, Ensemble, ScoreEstimator, LikelihoodEstimator, load_estimator
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
        hist_bins=None,
        include_xsec=True,
        thetaref=None,
        resolutions=25,
        luminosity=300000.0,
        n_histo_toys=None,
        returns="pval",
        dof=None,
        n_observed=None,
        histo_theta_batchsize=100,
        weighted_histo=True,
        score_components=None,
        test_split=0.2,
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
            n_histo_toys,
            returns=returns,
            dof=dof,
            histo_theta_batchsize=histo_theta_batchsize,
            weighted_histo=weighted_histo,
            score_components=score_components,
            test_split=test_split,
            thetaref=thetaref,
        )
        return theta_grid, return_values, i_ml

    def expected_limits(
        self,
        theta_true,
        theta_ranges,
        mode="ml",
        model_file=None,
        hist_vars=None,
        hist_bins=None,
        include_xsec=True,
        thetaref=None,
        resolutions=25,
        luminosity=300000.0,
        n_histo_toys=None,
        returns="pval",
        dof=None,
        histo_theta_batchsize=100,
        weighted_histo=True,
        score_components=None,
        sample_only_from_closest_benchmark=True,
        test_split=0.2,
    ):
        logger.info("Generating Asimov data")
        x_asimov, x_weights = self._asimov_data(
            theta_true, sample_only_from_closest_benchmark=sample_only_from_closest_benchmark, test_split=test_split
        )
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
            n_histo_toys,
            returns=returns,
            dof=dof,
            histo_theta_batchsize=histo_theta_batchsize,
            theta_true=theta_true,
            weighted_histo=weighted_histo,
            score_components=score_components,
            test_split=test_split,
            thetaref=thetaref,
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
        hist_bins=None,
        include_xsec=True,
        obs_weights=None,
        luminosity=300000.0,
        n_histo_toys=10000,
        returns="pval",
        dof=None,
        histo_theta_batchsize=100,
        theta_true=None,
        weighted_histo=True,
        score_components=None,
        test_split=0.2,
        thetaref=None,
    ):
        logger.debug("Calculating p-values for %s expected events", n_events)

        # Inputs
        assert returns in ["pval", "llr", "llr_raw"], "returns has to be either 'pval','llr' or 'llr_raw'!"

        if thetaref is None and mode in ["sallino", "adaptive-sally"]:
            thetaref = np.zeros(self.n_parameters)
            logging.warning(
                "The SALLINO and adaptive SALLY methods require the reference point, but the argument thetaref was not"
                " provided. Assuming thetaref = %s.",
                thetaref,
            )

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
            log_r_kin = self._calculate_log_likelihood_ratio_kinematics(x, theta_grid, model, theta_true)
            log_r_kin = log_r_kin.astype(np.float64)
            log_r_kin = self._clean_nans(log_r_kin)
            logger.debug("Raw mean -2 log r: %s", np.mean(-2.0 * log_r_kin, axis=1))
            log_r_kin = n_events * np.sum(log_r_kin * obs_weights[np.newaxis, :], axis=1)
            logger.debug("Rescaled -2 log r: %s", -2.0 * log_r_kin)

        elif mode in ["histo", "sally", "adaptive-sally", "sallino"]:
            if mode == "histo" and hist_vars is None:
                logger.warning(
                    "SALLY inference with mode='histo' is deprecated. Please use mode='sally', "
                    "mode='adaptive-sally', or mode='sallino' instead."
                )
                mode = "sally"

            # Make summary statistic
            if mode == "histo":
                assert hist_vars is not None
                logger.info("Setting up standard summary statistics")
                summary_function = self._make_summary_statistic_function("observables", observables=hist_vars)
                processor = None

            elif mode in ["sally", "adaptive-sally", "sallino"]:
                if score_components is None:
                    logger.info("Loading score estimator and setting all components up as summary statistics")
                else:
                    logger.info(
                        "Loading score estimator and setting components %s up as summary statistics", score_components
                    )
                model = load_estimator(model_file)
                summary_function = self._make_summary_statistic_function(
                    "sally", model=model, observables=score_components
                )
                processor = self._make_score_processor(mode, thetaref=thetaref)

            else:
                raise RuntimeError("For 'histo' mode, either provide histo_vars or model_file!")

            # Calculate summary stats (before SALLINO / adaptive-SALLY transforms)
            summary_stats = summary_function(x)
            del x

            # Dimension of summary statistic space
            n_summary_stats = summary_stats.shape[1]
            if mode == "adaptive-sally":
                n_summary_stats += 1
            elif mode == "sallino":
                n_summary_stats = 1

            # Bin numbers
            if hist_bins is None:
                if mode == "adaptive-sally":
                    hist_bins = tuple([20] + [5 for _ in range(n_summary_stats - 1)])
                    total_n_bins = 20 * 5 ** (n_summary_stats - 1)
                elif n_summary_stats == 1:
                    hist_bins = 50
                    total_n_bins = 50
                elif n_summary_stats == 2:
                    hist_bins = 10
                    total_n_bins = 10 ** 2
                else:
                    hist_bins = 5
                    total_n_bins = 5 ** n_summary_stats
            elif isinstance(hist_bins, int):
                total_n_bins = hist_bins ** n_summary_stats
            else:
                total_n_bins = np.prod(hist_bins)

            logger.info(
                "Creating histograms of %s summary statistics. Using %s bins each, or %s in total.",
                n_summary_stats,
                hist_bins,
                total_n_bins,
            )
            histos = self._make_histos(
                summary_function,
                hist_bins,
                theta_grid,
                n_histo_toys,
                histo_theta_batchsize=histo_theta_batchsize,
                weighted_histo=weighted_histo,
                test_split=test_split,
                processor=processor,
            )

            logger.info("Calculating kinematic log likelihood with histograms")
            log_r_kin = self._calculate_log_likelihood_histo(summary_stats, theta_grid, histos, processor=processor)
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
        log_r, i_ml = self._subtract_mle(log_r)
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
                if observables is not None:
                    score = score[:, observables]
                return score

        else:
            raise RuntimeError("Unknown mode {}, has to be 'observables' or 'sally'".format(mode))

        return summary_function

    def _make_score_processor(self, mode, thetaref, epsilon=1.0e-9):

        """
        From old project code:

        # Delta_theta
        delta_theta = theta - settings.thetas[theta1]
        delta_theta_norm = np.linalg.norm(delta_theta)
        if delta_theta_norm > settings.epsilon:
            rotation_matrix = (np.array([[delta_theta[0], - delta_theta[1]], [delta_theta[1], delta_theta[0]]])
                               / np.linalg.norm(delta_theta))
        else:
            rotation_matrix = np.identity(2)

        # Prepare calibration data
        tthat_calibration = that_calibration.dot(delta_theta)
        that_rotated_calibration = that_calibration.dot(rotation_matrix)
        """

        if mode == "adaptive-score":

            def processor(scores, theta):
                delta_theta = theta - thetaref
                h = scores.dot(delta_theta.flatten()).reshape((-1, 1))
                return np.concatenate((h, scores), axis=0)

        elif mode == "sallino":

            def processor(scores, theta):
                delta_theta = theta - thetaref
                h = scores.dot(delta_theta.flatten()).reshape((-1, 1))
                return h

        else:

            def processor(scores, theta):
                return scores

        return processor

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

    def _asimov_data(self, theta, test_split=0.2, sample_only_from_closest_benchmark=True):
        start_event, end_event, correction_factor = self._train_test_split(False, test_split)
        x, weights_benchmarks = next(
            self.event_loader(
                start=start_event,
                end=end_event,
                batch_size=None,
                generated_close_to=theta if sample_only_from_closest_benchmark else None,
            )
        )
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

    def _make_histos(
        self,
        summary_function,
        x_bins,
        theta_grid,
        n_histo_toys=1000,
        histo_theta_batchsize=100,
        weighted_histo=True,
        test_split=0.2,
        processor=None,
    ):
        if weighted_histo:
            logger.debug("Generating weighted histo data")
            summary_stats, all_weights = self._make_weighted_histo_data(
                summary_function, theta_grid, n_histo_toys, test_split=test_split
            )

            logger.debug("Making histograms")
            histos = []
            for theta, weights in zip(theta_grid, all_weights):
                if processor is None:
                    data = summary_stats
                else:
                    data = processor(summary_stats, theta)
                histos.append(Histo(data, weights, x_bins, fill_empty=1.0e-3))

        else:
            logger.debug("Generating sampled histo data and making histograms")
            histos = []

            n_thetas = len(theta_grid)
            n_batches = (n_thetas - 1) // histo_theta_batchsize + 1
            for i_batch in range(n_batches):
                logger.debug("Generating histogram data for batch %s / %s", i_batch + 1, n_batches)
                theta_batch = theta_grid[i_batch * histo_theta_batchsize : (i_batch + 1) * histo_theta_batchsize]

                _, all_summary_stats = self._make_sampled_histo_data(
                    summary_function, theta_batch, n_histo_toys, test_split=test_split
                )
                for theta, summary_stats in zip(theta_batch, all_summary_stats):
                    if processor is None:
                        data = summary_stats
                    else:
                        data = processor(summary_stats, theta)
                    histos.append(Histo(data, weights=None, bins=x_bins, fill_empty=1.0e-3))

        return histos

    def _make_weighted_histo_data(self, summary_function, thetas, n_toys, test_split=0.2):
        # Get weighted events
        start_event, end_event, _ = self._train_test_split(True, test_split)
        x, weights_benchmarks = self.weighted_events(start_event=start_event, end_event=end_event, n_draws=n_toys)

        # Calculate summary stats
        summary_stats = summary_function(x)

        # Calculate weights for thetas
        weights = self._weights(thetas, None, weights_benchmarks)

        return summary_stats, weights

    def _make_sampled_histo_data(self, summary_function, thetas, n_toys_per_theta, test_split=0.2):
        sampler = SampleAugmenter(self.madminer_filename, include_nuisance_parameters=self.include_nuisance_parameters)

        if n_toys_per_theta is None:
            n_toys_per_theta = 10000

        x, theta, _ = sampler.sample_train_plain(
            theta=sampling.morphing_points(thetas),
            n_samples=n_toys_per_theta * len(thetas),
            test_split=test_split,
            filename=None,
            folder=None,
            suppress_logging=True,
        )

        summary_stats = summary_function(x)
        summary_stats = summary_stats.reshape((len(thetas), n_toys_per_theta, -1))

        return summary_stats

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
    def _calculate_log_likelihood_histo(summary_stats, theta_grid, histos, processor=None):
        log_p = []
        for theta, histo in zip(theta_grid, histos):
            if processor is None:
                data = summary_stats
            else:
                data = processor(summary_stats, theta)
            log_p.append(histo.log_likelihood(data))
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
        elif isinstance(model, LikelihoodEstimator):
            log_r, _ = model.evaluate_log_likelihood(
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
        elif isinstance(model, Ensemble) and model.estimator_type == "likelihood":
            log_r, _ = model.evaluate_log_likelihood(
                x=x_observed,
                theta=theta_grid,
                test_all_combinations=True,
                evaluate_score=False,
                calculate_covariance=False,
            )
        else:
            raise NotImplementedError(
                "Likelihood ratio estimation is currently only implemented for "
                "ParameterizedRatioEstimator and LikelihoodEstimator instancees"
            )
        return log_r

    @staticmethod
    def _subtract_mle(log_r):
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
