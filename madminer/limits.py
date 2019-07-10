from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
from scipy.stats import chi2, poisson

from madminer.analysis import DataAnalyzer
from madminer.utils.various import mdot, less_logging
from madminer.ml import ParameterizedRatioEstimator, Ensemble, ScoreEstimator, LikelihoodEstimator, load_estimator
from madminer.utils.histo import Histo
from madminer.sampling import SampleAugmenter
from madminer import sampling

logger = logging.getLogger(__name__)


class AsymptoticLimits(DataAnalyzer):
    """
    Statistical inference based on asymptotic properties of the likelihood ratio as
    test statistics.

    This class provides two high-level functions:

    * `AsymptoticLimits.observed_limits()` calculates p-values over a grid in parameter space for a given set of
      observed data.
    * `AsymptoticLimits.expected_limits()` calculates expected p-values over a grid in parameter space based on
      "Asimov data", a large hypothetical data set drawn from a given parameter point. This method is typically used
      to define expected exclusion limits or significances.

    Both functions support inference based on...

    * histograms of kinematic observables,
    * based on histograms of score vectors estimated with the `madminer.ml.ScoreEstimator` class (SALLY and SALLINO
      techniques),
    * based on likelihood or likelihood ratio functions estimated with the `madminer.ml.LikelihoodEstimator` and
      `madminer.ml.ParameterizedRatioEstimator` classes (NDE, SCANDAL, CARL, RASCAL, ALICES, and so on).

    Currently, this class requires a morphing setup. It does *not* yet support nuisance parameters.

    Parameters
    ----------
    filename : str
        Path to MadMiner file (for instance the output of `madminer.delphes.DelphesProcessor.save()`).

    include_nuisance_parameters : bool, optional
        If True, nuisance parameters are taken into account. Currently not implemented. Default value: False.
    """

    def __init__(self, filename=None, include_nuisance_parameters=False):
        if include_nuisance_parameters:
            raise NotImplementedError("AsymptoticLimits does not yet support nuisance parameters.")

        super(AsymptoticLimits, self).__init__(filename, False, include_nuisance_parameters=False)

    def observed_limits(
        self,
        mode,
        x_observed,
        grid_ranges,
        grid_resolutions=25,
        include_xsec=True,
        model_file=None,
        hist_vars=None,
        score_components=None,
        hist_bins=None,
        thetaref=None,
        luminosity=300000.0,
        weighted_histo=True,
        n_histo_toys=100000,
        histo_theta_batchsize=1000,
        n_observed=None,
        dof=None,
        test_split=0.2,
        return_histos=True,
        fix_adaptive_binning="auto",
    ):
        """
        Calculates p-values over a grid in parameter space based on a given set of observed events.

        `x_observed` specifies the observed data as an array of observables, using the same observables and their order
        as used throughout the MadMiner workflow.

        The p-values with frequentist hypothesis tests using the likelihood ratio as test statistic. The asymptotic
        approximation is used, see https://arxiv.org/abs/1007.1727.

        Depending on the keyword `mode`, the likelihood ratio is calculated with one of several different methods:

        * With `mode="rate"`, MadMiner only calculates the Poisson likelihood of the total number of events.
        * With `mode="histo"`, the kinematic likelihood is estimated with histograms of a small number of observables
          given by the keyword `hist_vars`. `hist_bins` determines the binning of the histograms. `include_xsec`
          sets whether the Poisson likelihood of the total number of events is included or not.
        * With `mode="ml"`, the likelihood ratio is estimated with a parameterized neural network. `model_file` has to
          point to the filename of a saved `LikelihoodEstimator` or `ParameterizedRatioEstimator` instance or a
          corresponding `Ensemble` (i.e. be the same filename used when calling `estimator.save()`). `include_xsec`
          sets whether the Poisson likelihood of the total number of events is included or not.
        * With `mode="sally"`, the likelihood ratio is estimated with histograms of the components of the estimated
          score vector. `model_file` has to point to the filename of a saved `ScoreEstimator` instance. With
          `score_components`, the histogram can be restricted to some components of the score. `hist_bins` defines the
          binning of the histograms. `include_xsec`
          sets whether the Poisson likelihood of the total number of events is included or not.
        * With `mode="adaptive-sally"`, the likelihood ratio is estimated with histograms of the components of the
          estimated score vector. The approach is essentially the same as for `"sally"`, but the histogram binning is
          optimized for every parameter point by adding a new `h = score * (theta - thetaref)` dimension to the
          histogram. `include_xsec`
          sets whether the Poisson likelihood of the total number of events is included or not.
        * With `mode="sallino"`, the likelihood ratio is estimated with one-dimensional histograms of the scalar
          variable `h = score * (theta - thetaref)` for each point `theta` along the parameter grid. `model_file` has to
          point to the filename of a saved `ScoreEstimator` instance.  `hist_bins` defines the binning of the histogram.
          `include_xsec` sets whether the Poisson likelihood of the total number of events is included or not.

        MadMiner calculates one p-value for every parameter point on an evenly spaced grid specified by `grid_ranges`
        and `grid_resolutions`. For instance, in a three-dimensional parameter space,
        `grid_ranges=[(-1., 1.), (-2., 2.), (-3., 3.)]` and `grid_resolutions=[10,10,10]` will start the calculation along
        10^3 parameter points in a cube with edges `(-1, 1)` in the first parameter and so on.

        Parameters
        ----------
        mode : {"rate", "histo", "ml", "sally", "sallino", "adaptive-sally"}
            Defines how the likelihood ratio test statistic is calculated. See above.

        x_observed : ndarray
            Observed data with shape `(n_events, n_observables)`. The observables have to be the same used throughout
            the MadMiner analysis, for instance specified in the `DelphesReader` class with `add_observables`.

        grid_ranges : list of tuple of float
            Specifies the boundaries of the parameter grid on which the p-values are evaluated. It should be
            `[(min, max), (min, max), ..., (min, max)]`, where the list goes over all parameters and `min` and `max` are
            float.

        grid_resolutions : int or list of int, optional
            Resolution of the parameter space grid on which the p-values are evaluated. If int, the resolution is the
            same along every dimension of the hypercube. If list of int, the individual entries specify the number of
            points along each parameter individually. Default value: 25.

        include_xsec : bool, optional
            Whether the Poisson likelihood representing the total number of events is included in the analysis.
            Default value: True.

        model_file : str or None, optional
            Filename of a saved neural network estimating the likelihood, likelihood ratio, or score. Required if
            mode is anything except "rate" or "histo". Default value: None.

        hist_vars : list of str or None, optional
            Kinematic variables used in the histograms when mode is "histo". The names are the same as used for instance
            in `DelphesReader`. Default value: None.

        score_components : None or list of int, optional
            Defines the score components used when mode is "sally" or "adaptive-sally". Default value: None.

        hist_bins : int or list of (int or ndarray) or None, optional
            Defines the histogram binning when mode is "histo", "sally", "adaptive-sally", or "sallino". If int, gives
            the number of bins automatically chosen for each summary statistic. If list, each entry corresponds to one
            summary statistic (e.g. kinematic variable specified by hist_vars or estimated score component); an int
            entry corresponds to the number of automatically chosen bins, an ndarray specifies the bin edges along
            this dimension explicitly. If None, the bins are chosen according to the defaults, which depend on mode and
            the number of summary statistics. When mode is "adaptive-sally", the first summary statistic is
            `h = score * (theta - thetaref)`, the remaining ones are the score components. Default value: None.

        thetaref : ndarray or None, optional
            Defines the reference parameter point at which the score is evaluated for mode "sallino" or
            "adaptive-sally". If None, the origin in parameter space, [0., 0., ..., 0.], is used. Default value: None.

        luminosity : float, optional
            Integrated luminosity in pb^{-1} assumed in the analysis. Default value: 300000.

        weighted_histo : bool, optional
            If True, the histograms used for the modes "histo", "sally", "sallino", and "adaptive-sally" use one set of
            weighted events to construct the histograms at every point along the parameter grid, only with different
            weights for each parameter point on the grid. If False, independent unweighted event samples are drawn for
            each parameter point on the grid. Default value: True.

        n_histo_toys : int or None, optional
            Number of events drawn to construct the histograms used for the modes "histo", "sally", "sallino", and
            "adaptive-sally". If None and weighted_histo is True, all events in the training fraction of the MadMiner
            file are used. If None and weighted_histo is False, 100000 events are used. Default value: 100000.

        histo_theta_batchsize : int or None, optional
            Number of histograms constructed in parallel for the modes "histo", "sally", "sallino", and "adaptive-sally"
            and if weighted_histo is True. A larger number speeds up the calculation, but requires more memory.
            Default value: 1000.

        n_observed : int or None, optional
            If not None, the likelihood ratio is rescaled to this number of observed events before calculating p-values.
            Default value: None.

        dof : int or None, optional
            If not None, sets the number of parameters for the calculation of the p-values. If None, the overall number
            of parameters is used. Default value: None.

        test_split : float, optional
            Fraction of weighted events in the MadMiner file reserved for evaluation. Default value: 0.2.

        return_histos : bool, optional
            If True and if mode is "histo", "sally", "adaptive-sally", or "sallino", the function returns histogram
            objects for each point along the grid.

        fix_adaptive_binning : bool or "auto", optional
            If True and if mode is "histo", "sally", "adaptive-sally", or "sallino", the automatic histogram binning
            is the same for every point along the parameter grid. If "auto", this option is turned on if mode is
            "histo" or "sally", but not for "adaptive-sally" or "sallino". Default value: "auto".

        Returns
        -------
        parameter_grid : ndarray
            Parameter points at which the p-values are evaluated with shape `(n_grid_points, n_parameters)`.

        p_values : ndarray
            Observed p-values for each parameter point on the grid, with shape `(n_grid_points,)`.

        mle : int
            Index of the parameter point with the best fit (largest p-value / smallest -2 log likelihood ratio).

        log_likelihood_ratio_kin : ndarray or None
            log likelihood ratio based only on kinematics for each point of the grid, with shape `(n_grid_points,)`.

        log_likelihood_rate : ndarray or None
            log likelihood based only on the total rate for each point of the grid, with shape `(n_grid_points,)`.

        histos : None or list of Histogram
            None if return_histos is False. Otherwise a list of histogram objects for each point on the grid. This
            can be useful for debugging or for plotting the histograms.

        """
        if n_observed is None:
            n_observed = len(x_observed)
        results = self._analyse(
            n_observed,
            x_observed,
            grid_ranges,
            grid_resolutions,
            mode,
            model_file,
            hist_vars,
            hist_bins,
            include_xsec,
            None,
            luminosity,
            n_histo_toys,
            return_histos=return_histos,
            dof=dof,
            histo_theta_batchsize=histo_theta_batchsize,
            weighted_histo=weighted_histo,
            score_components=score_components,
            test_split=test_split,
            thetaref=thetaref,
            fix_adaptive_binning=fix_adaptive_binning,
        )
        return results

    def expected_limits(
        self,
        mode,
        theta_true,
        grid_ranges,
        grid_resolutions=25,
        include_xsec=True,
        model_file=None,
        hist_vars=None,
        score_components=None,
        hist_bins=None,
        thetaref=None,
        luminosity=300000.0,
        weighted_histo=True,
        n_histo_toys=100000,
        histo_theta_batchsize=1000,
        dof=None,
        test_split=0.2,
        return_histos=True,
        fix_adaptive_binning="auto",
        sample_only_from_closest_benchmark=True,
    ):

        """
        Calculates expected p-values over a grid in parameter space.

        `theta_true` specifies which parameter point is assumed to be true. Based on this parameter point, the function
        generates a large artificial "Asimov data set". p-values are then calculated with frequentist hypothesis tests
        using the likelihood ratio as test statistic. The asymptotic approximation is used, see
        https://arxiv.org/abs/1007.1727.

        Depending on the keyword `mode`, the likelihood ratio is calculated with one of several different methods:

        * With `mode="rate"`, MadMiner only calculates the Poisson likelihood of the total number of events.
        * With `mode="histo"`, the kinematic likelihood is estimated with histograms of a small number of observables
          given by the keyword `hist_vars`. `hist_bins` determines the binning of the histograms. `include_xsec`
          sets whether the Poisson likelihood of the total number of events is included or not.
        * With `mode="ml"`, the likelihood ratio is estimated with a parameterized neural network. `model_file` has to
          point to the filename of a saved `LikelihoodEstimator` or `ParameterizedRatioEstimator` instance or a
          corresponding `Ensemble` (i.e. be the same filename used when calling `estimator.save()`). `include_xsec`
          sets whether the Poisson likelihood of the total number of events is included or not.
        * With `mode="sally"`, the likelihood ratio is estimated with histograms of the components of the estimated
          score vector. `model_file` has to point to the filename of a saved `ScoreEstimator` instance. With
          `score_components`, the histogram can be restricted to some components of the score. `hist_bins` defines the
          binning of the histograms. `include_xsec`
          sets whether the Poisson likelihood of the total number of events is included or not.
        * With `mode="adaptive-sally"`, the likelihood ratio is estimated with histograms of the components of the
          estimated score vector. The approach is essentially the same as for `"sally"`, but the histogram binning is
          optimized for every parameter point by adding a new `h = score * (theta - thetaref)` dimension to the
          histogram. `include_xsec`
          sets whether the Poisson likelihood of the total number of events is included or not.
        * With `mode="sallino"`, the likelihood ratio is estimated with one-dimensional histograms of the scalar
          variable `h = score * (theta - thetaref)` for each point `theta` along the parameter grid. `model_file` has to
          point to the filename of a saved `ScoreEstimator` instance.  `hist_bins` defines the binning of the histogram.
          `include_xsec` sets whether the Poisson likelihood of the total number of events is included or not.

        MadMiner calculates one p-value for every parameter point on an evenly spaced grid specified by `grid_ranges`
        and `grid_resolutions`. For instance, in a three-dimensional parameter space,
        `grid_ranges=[(-1., 1.), (-2., 2.), (-3., 3.)]` and `grid_resolutions=[10,10,10]` will start the calculation
        along 10^3 parameter points in a cube with edges `(-1, 1)` in the first parameter and so on.

        Parameters
        ----------
        mode : {"rate", "histo", "ml", "sally", "sallino", "adaptive-sally"}
            Defines how the likelihood ratio test statistic is calculated. See above.

        theta_true : ndarray
            Parameter point assumed to be true to calculate the Asimov data.

        grid_ranges : list of tuple of float
            Specifies the boundaries of the parameter grid on which the p-values are evaluated. It should be
            `[(min, max), (min, max), ..., (min, max)]`, where the list goes over all parameters and `min` and `max` are
            float.

        grid_resolutions : int or list of int, optional
            Resolution of the parameter space grid on which the p-values are evaluated. If int, the resolution is the
            same along every dimension of the hypercube. If list of int, the individual entries specify the number of
            points along each parameter individually. Default value: 25.

        include_xsec : bool, optional
            Whether the Poisson likelihood representing the total number of events is included in the analysis.
            Default value: True.

        model_file : str or None, optional
            Filename of a saved neural network estimating the likelihood, likelihood ratio, or score. Required if
            mode is anything except "rate" or "histo". Default value: None.

        hist_vars : list of str or None, optional
            Kinematic variables used in the histograms when mode is "histo". The names are the same as used for instance
            in `DelphesReader`. Default value: None.

        score_components : None or list of int, optional
            Defines the score components used when mode is "sally" or "adaptive-sally". Default value: None.

        hist_bins : int or list of (int or ndarray) or None, optional
            Defines the histogram binning when mode is "histo", "sally", "adaptive-sally", or "sallino". If int, gives
            the number of bins automatically chosen for each summary statistic. If list, each entry corresponds to one
            summary statistic (e.g. kinematic variable specified by hist_vars or estimated score component); an int
            entry corresponds to the number of automatically chosen bins, an ndarray specifies the bin edges along
            this dimension explicitly. If None, the bins are chosen according to the defaults, which depend on mode and
            the number of summary statistics. When mode is "adaptive-sally", the first summary statistic is
            `h = score * (theta - thetaref)`, the remaining ones are the score components. Default value: None.

        thetaref : ndarray or None, optional
            Defines the reference parameter point at which the score is evaluated for mode "sallino" or
            "adaptive-sally". If None, the origin in parameter space, [0., 0., ..., 0.], is used. Default value: None.

        luminosity : float, optional
            Integrated luminosity in pb^{-1} assumed in the analysis. Default value: 300000.

        weighted_histo : bool, optional
            If True, the histograms used for the modes "histo", "sally", "sallino", and "adaptive-sally" use one set of
            weighted events to construct the histograms at every point along the parameter grid, only with different
            weights for each parameter point on the grid. If False, independent unweighted event samples are drawn for
            each parameter point on the grid. Default value: True.

        n_histo_toys : int or None, optional
            Number of events drawn to construct the histograms used for the modes "histo", "sally", "sallino", and
            "adaptive-sally". If None and weighted_histo is True, all events in the training fraction of the MadMiner
            file are used. If None and weighted_histo is False, 100000 events are used. Default value: 100000.

        histo_theta_batchsize : int or None, optional
            Number of histograms constructed in parallel for the modes "histo", "sally", "sallino", and "adaptive-sally"
            and if weighted_histo is True. A larger number speeds up the calculation, but requires more memory.
            Default value: 1000.

        dof : int or None, optional
            If not None, sets the number of parameters for the calculation of the p-values. If None, the overall number
            of parameters is used. Default value: None.

        test_split : float, optional
            Fraction of weighted events in the MadMiner file reserved for evaluation. Default value: 0.2.

        return_histos : bool, optional
            If True and if mode is "histo", "sally", "adaptive-sally", or "sallino", the function returns histogram
            objects for each point along the grid.

        fix_adaptive_binning : bool or "auto", optional
            If True and if mode is "histo", "sally", "adaptive-sally", or "sallino", the automatic histogram binning
            is the same for every point along the parameter grid. If "auto", this option is turned on if mode is
            "histo" or "sally", but not for "adaptive-sally" or "sallino". Default value: "auto".

        sample_only_from_closest_benchmark : bool, optional
            If True, only events originally generated from the closest benchmarks are used when generating
            the Asimov data (and, if weighted_histo is False, the histogram data). Default value: True.

        Returns
        -------
        parameter_grid : ndarray
            Parameter points at which the p-values are evaluated with shape `(n_grid_points, n_parameters)`.

        p_values : ndarray
            Observed p-values for each parameter point on the grid, with shape `(n_grid_points,)`.

        mle : int
            Index of the parameter point with the best fit (largest p-value / smallest -2 log likelihood ratio).

        log_likelihood_ratio_kin : ndarray or None
            log likelihood ratio based only on kinematics for each point of the grid, with shape `(n_grid_points,)`.

        log_likelihood_rate : ndarray or None
            log likelihood based only on the total rate for each point of the grid, with shape `(n_grid_points,)`.

        histos : None or list of Histogram
            None if return_histos is False. Otherwise a list of histogram objects for each point on the grid. This
            can be useful for debugging or for plotting the histograms.

        """

        logger.info("Generating Asimov data")
        x_asimov, x_weights = self._asimov_data(
            theta_true, sample_only_from_closest_benchmark=sample_only_from_closest_benchmark, test_split=test_split
        )
        n_observed = luminosity * self._calculate_xsecs([theta_true])[0]
        logger.info("Expected events: %s", n_observed)
        results = self._analyse(
            n_observed,
            x_asimov,
            grid_ranges,
            grid_resolutions,
            mode,
            model_file,
            hist_vars,
            hist_bins,
            include_xsec,
            x_weights,
            luminosity,
            n_histo_toys,
            return_histos=return_histos,
            dof=dof,
            histo_theta_batchsize=histo_theta_batchsize,
            theta_true=theta_true,
            weighted_histo=weighted_histo,
            score_components=score_components,
            test_split=test_split,
            thetaref=thetaref,
            fix_adaptive_binning=fix_adaptive_binning,
        )
        return results

    def asymptotic_p_value(self, log_likelihood_ratio, dof=None):
        """
        Calculates the p-value corresponding to a given log likelihood ratio and number of degrees of freedom assuming
        the asymptotic approximation.

        Parameters
        ----------
        log_likelihood_ratio : ndarray
            Log likelihood ratio (without the factor -2)

        dof : int or None, optional
            Number of parameters / degrees of freedom. None means the overall number of parameters is used. Default
            value: None.

        Returns
        -------
        p_values : ndarray
            p-values.

        """
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
        mode,
        model_file=None,
        hist_vars=None,
        hist_bins=None,
        include_xsec=True,
        obs_weights=None,
        luminosity=300000.0,
        n_histo_toys=100000,
        return_histos=False,
        dof=None,
        histo_theta_batchsize=1000,
        theta_true=None,
        weighted_histo=True,
        score_components=None,
        test_split=0.2,
        thetaref=None,
        fix_adaptive_binning="auto",
    ):
        logger.debug("Calculating p-values for %s expected events", n_events)

        # Inputs
        if thetaref is None and mode in ["sallino", "adaptive-sally"]:
            thetaref = np.zeros(self.n_parameters)
            logger.warning(
                "The SALLINO and adaptive SALLY methods require the reference point, but the argument thetaref was not"
                " provided. Assuming thetaref = %s.",
                thetaref,
            )

        if fix_adaptive_binning == "auto":
            fix_adaptive_binning = mode in ["sally", "histo"]

        # Observation weights
        if obs_weights is None:
            obs_weights = np.ones(len(x))
        obs_weights /= np.sum(obs_weights)
        obs_weights = obs_weights.astype(np.float64)

        # Theta grid
        theta_grid, theta_middle = self._make_theta_grid(theta_ranges, theta_resolutions)

        histos = None

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
                processor = self._make_score_processor(mode, score_components=score_components, thetaref=thetaref)

            else:
                raise RuntimeError("For 'histo' mode, either provide histo_vars or model_file!")

            # Calculate summary stats (before SALLINO / adaptive-SALLY transforms)
            summary_stats = summary_function(x)
            del x

            # Dimension of summary statistic space
            hist_bins, n_bins_each, n_summary_stats, total_n_bins = self._find_bins(mode, hist_bins, summary_stats)

            # Make histograms
            logger.info(
                "Creating histograms of %s summary statistics. Using %s bins each, or %s in total.",
                n_summary_stats,
                n_bins_each,
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
                theta_binning=theta_middle if fix_adaptive_binning else None,
            )

            # Evaluate histograms
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

        logger.debug("Combined -2 log r: %s", -2.0 * log_r)
        log_r, i_ml = self._subtract_mle(log_r)
        logger.debug("Min-subtracted -2 log r: %s", -2.0 * log_r)
        p_values = self.asymptotic_p_value(log_r, dof=dof)

        return theta_grid, p_values, i_ml, log_r_kin, log_p_xsec, histos if return_histos else None

    def _find_bins(self, mode, hist_bins, summary_stats):
        n_summary_stats = summary_stats.shape[1]
        if mode == "adaptive-sally":
            n_summary_stats += 1
        elif mode == "sallino":
            n_summary_stats = 1
        # Bin numbers
        if hist_bins is None:
            if mode == "adaptive-sally":
                hist_bins = tuple([12] + [5 for _ in range(n_summary_stats - 1)])
                total_n_bins = 12 * 5 ** (n_summary_stats - 1)
            elif n_summary_stats == 1:
                hist_bins = 25
                total_n_bins = 25
            elif n_summary_stats == 2:
                hist_bins = 8
                total_n_bins = 8 ** 2
            else:
                hist_bins = 5
                total_n_bins = 5 ** n_summary_stats
            n_bins_each = hist_bins
        elif isinstance(hist_bins, int):
            total_n_bins = hist_bins ** n_summary_stats
            n_bins_each = hist_bins
        else:
            n_bins_each = [n_bins if isinstance(n_bins, int) else len(n_bins) - 1 for n_bins in hist_bins]
            total_n_bins = np.prod(n_bins_each)
        return hist_bins, n_bins_each, n_summary_stats, total_n_bins

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

    def _make_score_processor(self, mode, score_components, thetaref, epsilon=1.0e-6):
        if mode == "adaptive-sally":

            def processor(scores, theta):
                delta_theta = theta - thetaref
                if score_components is not None:
                    delta_theta = delta_theta[score_components]
                if np.linalg.norm(delta_theta) > epsilon:
                    h = scores.dot(delta_theta.flatten()).reshape((-1, 1)) / np.linalg.norm(delta_theta)
                else:
                    h = scores[:, 0]
                h = h.reshape((-1, 1))
                return np.concatenate((h, scores), axis=1)

        elif mode == "sallino":

            def processor(scores, theta):
                delta_theta = theta - thetaref
                if score_components is not None:
                    delta_theta = delta_theta[score_components]
                if np.linalg.norm(delta_theta) > epsilon:
                    h = scores.dot(delta_theta.flatten()).reshape((-1, 1)) / np.linalg.norm(delta_theta)
                else:
                    h = scores[:, 0]
                h = h.reshape((-1, 1))
                return h

        elif mode == "sally":

            def processor(scores, theta):
                return scores

        else:
            raise RuntimeError("Unknown score processing mode {}".format(mode))

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
        theta_middle = []
        for resolution, (theta_min, theta_max) in zip(resolutions, theta_ranges):
            theta_each.append(np.linspace(theta_min, theta_max, resolution))
            theta_middle.append(0.5 * (theta_max + theta_min))
        theta_grid_each = np.meshgrid(*theta_each, indexing="ij")
        theta_grid_each = [theta.flatten() for theta in theta_grid_each]
        theta_grid = np.vstack(theta_grid_each).T
        theta_middle = np.asarray(theta_middle)
        return theta_grid, theta_middle

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
        theta_binning=None,
    ):
        if theta_binning is not None:
            logger.info("Determining fixed adaptive histogram binning for theta = %s", theta_binning)
            x_bins = self._fixed_adaptive_binning(
                n_histo_toys, processor, summary_function, test_split, theta_binning, x_bins
            )
            logger.debug("Fixed adaptive binning: %s", x_bins)

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
                histos.append(Histo(data, weights, x_bins, fill_empty=1.0e-9))

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
                    histos.append(Histo(data, weights=None, bins=x_bins, fill_empty=1.0e-9))

        return histos

    def _fixed_adaptive_binning(self, n_histo_toys, processor, summary_function, test_split, theta_binning, x_bins):
        summary_stats, [weights] = self._make_weighted_histo_data(
            summary_function, [theta_binning], n_histo_toys, test_split=test_split
        )
        if processor is None:
            data = summary_stats
        else:
            data = processor(summary_stats, theta_binning)
        histo = Histo(data, weights, x_bins, fill_empty=1.0e-9)
        x_bins = histo.edges
        return x_bins

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
            n_toys_per_theta = 100000

        with less_logging():
            x, theta, _ = sampler.sample_train_plain(
                theta=sampling.morphing_points(thetas),
                n_samples=n_toys_per_theta * len(thetas),
                test_split=test_split,
                filename=None,
                folder=None,
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
