from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
import time
from scipy.stats import poisson, norm, chi2
from scipy.optimize import minimize
from itertools import product

from madminer.analysis import DataAnalyzer
from madminer.utils.various import mdot, less_logging, math_commands
from madminer.ml import ParameterizedRatioEstimator, ScoreEstimator, Ensemble, LikelihoodEstimator, load_estimator
from madminer.utils.histo import Histo
from madminer.sampling import SampleAugmenter
from madminer import sampling

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


class HistoLikelihood(BaseLikelihood):
    def create_negative_log_likelihood(
        self,
        x_observed,
        observables=None,
        score_components=None,
        n_observed=None,
        x_observed_weights=None,
        include_xsec=True,
        luminosity=300000.0,
        mode="sampled",
        n_histo_toys=100000,
        model_file=None,
        hist_bins=None,
        thetas_binning=None,
        test_split=None,
    ):
        """
        Returns a function which calculates the negative log likelihood for a given
        parameter point, evaulated with a dataset (x_observed,n_observed,x_observed_weights).
            
        Parameters
        ----------
            
        x_observed : list of ndarray
            Set of event observables with shape `(n_events, n_observables)`.
            
        observables : list of str or None , optional
            Kinematic variables used in the histograms. The names are the same as
            used for instance in `DelphesReader`.
            
        score_components : None or list of int, optional
            Defines the score components used. Default value: None.
            
        n_observed : int or None , optional
            If int, number of observed events. If None, n_observed is defined by
            the length of x_observed. Default: None.
            
        x_observed_weights : list of float or None , optional
            List of event weights with shape `(n_events)`. If None, all events have equal
            weights. Default: None.
            
        include_xsec : bool, optional
            Whether the Poisson likelihood representing the total number of events is
            included in the analysis. Default value: True.
            
        luminosity : float, optional
            Integrated luminosity in pb^{-1} assumed in the analysis. Default value: 300000.
            
        mode : {"weighted" , "sampled", "histo"} , optional
            If "sampled", for each evaulation of the likelihood function, a separate
            set of events are sampled and histogram is created to construct the
            likelihood function. If "weighted", first a set of weighted events is
            sampled which is then used to create histograms. Default value: "sampled"
            
        n_histo_toys : int or None, optional
            Number of events drawn to construct the histograms used. If None and weighted_histo
            is True, all events in the training fraction of the MadMiner file are used. If None
            and weighted_histo is False, 100000 events are used. Default value: 100000.
            
        model_file : str or None, optional
            Filename of a saved neural network estimating the score. Required if
            score_components is not None. Default value: None.
            
        hist_bins : int or list of (int or ndarray) or None, optional
            Defines the histogram binning. If int, gives the number of bins automatically
            chosen for each summary statistic. If list, each entry corresponds to one
            summary statistic (e.g. kinematic variable specified by hist_vars); an int
            entry corresponds to the number of automatically chosen bins, an ndarray specifies
            the bin edges along this dimension explicitly. If None, the bins are chosen according
            to the defaults: for one summary statistic the default is 25 bins, for 2 it's 8 bins
            along each direction, for more it's 5 per dimension. Default value: None.
            
        thetas_binning : list of ndarray or None
            Specifies the parameter points used to determine the optimal binning.
            This is requires if hist_bins doesn't already fully specify the
            binning of the histogram. Default value : None
            
        test_split :
            
        Returns
        -------
        negative_log_likelihood : likelihood
            Function that evaluates the negative log likelihood for a given parameter point
            
        """

        # Check input and join observables and score components - nothing interesting
        if observables is None:
            observables = list([])
        if score_components is None:
            score_components = list([])
        if observables == [] and score_components == []:
            logger.info("No observables and scores provided. Calculate LLR due to rate and set include_xsec=True.")
            include_xsec = True
        observables = list(observables) + list(score_components)

        if n_observed is None:
            n_observed = len(x_observed)

        supported_modes = ["sampled", "weighted", "histo"]
        if mode not in supported_modes:
            raise ValueError("Mode %s unknown. Choose one of the following methods: %s", mode, supported_modes)

        # Load model - nothing interesting
        if score_components != []:
            assert all([isinstance(score_component, int) for score_component in score_components])
            if model_file is None:
                raise ValueError("You need to provide a model_file!")
            model = load_estimator(model_file)
        else:
            model = None

        # Create summary function
        logger.info("Setting up standard summary statistics")
        summary_function = None
        if observables != []:
            summary_function = self._make_summary_statistic_function(observables=observables, model=model)

        # Weighted sampled
        data, summary_stats, weights_benchmarks = None, None, None
        if mode == "weighted" or mode == "histo":
            logger.info("Getting weighted data")
            data, weights_benchmarks = self._make_histo_data_weighted(
                summary_function=summary_function, n_toys=n_histo_toys, test_split=test_split
            )
        if (mode == "weighted" or mode == "histo") and observables != []:
            summary_stats = summary_function(x_observed)

        # find binning
        logger.info("Setting up binning")
        if observables != [] and (
            hist_bins is None or not all([hasattr(hist_bin, "__len__") for hist_bin in hist_bins])
        ):
            if thetas_binning is None:
                raise ValueError("Your input requires adaptive binning: thetas_binning can not be None.")
            hist_bins = self._find_bins(hist_bins=hist_bins, n_summary_stats=len(observables))
            hist_bins = self._fixed_adaptive_binning(
                thetas_binning=thetas_binning,
                x_bins=hist_bins,
                data=data,
                weights_benchmarks=weights_benchmarks,
                n_toys=n_histo_toys,
                summary_function=summary_function,
            )
        logger.info("Use binning: %s", hist_bins)

        if mode == "histo":
            if hist_bins is not None:
                benchmark_histograms, _ = self._get_benchmark_histograms(data, weights_benchmarks, hist_bins)
                total_weights = np.array(
                    [sum(benchmark_histogram.flatten()) for benchmark_histogram in benchmark_histograms]
                )
            else:
                benchmark_histograms = None
                total_weights = np.array([sum(weights_benchmark) for weights_benchmark in weights_benchmarks.T])
        else:
            total_weights, benchmark_histograms = None, None

        # define negative likelihood function
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
                n_events=n_observed,
                xs=x_observed,
                theta=theta,
                nu=nu,
                include_xsec=include_xsec,
                luminosity=luminosity,
                x_weights=x_observed_weights,
                mode=mode,
                n_histo_toys=n_histo_toys,
                hist_bins=hist_bins,
                summary_function=summary_function,
                data=data,
                summary_stats=summary_stats,
                weights_benchmarks=weights_benchmarks,
                benchmark_histograms=benchmark_histograms,
                total_weights=total_weights,
            )
            return -log_likelihood

        return nll

    def create_expected_negative_log_likelihood(
        self,
        theta_true,
        nu_true,
        observables=None,
        score_components=None,
        include_xsec=True,
        luminosity=300000.0,
        n_asimov=None,
        mode="sampled",
        n_histo_toys=100000,
        model_file=None,
        hist_bins=None,
        thetas_binning=None,
        test_split=None,
    ):
        """
        Returns a function which calculates the expected negative log likelihood for a given
        parameter point, evaulated with test data sampled according to theta_true.
        
        Parameters
        ----------
        theta_true : ndarray
            Specifies the physical paramaters according to which the test data is sampled.
        
        nu_true : ndarray
            Specifies the nuisance paramaters according to which the test data is sampled.
        
        observables : list of str or None , optional
            Kinematic variables used in the histograms. The names are the same as
            used for instance in `DelphesReader`.
        
        score_components : None or list of int, optional
            Defines the score components used. Default value: None.
        
        include_xsec : bool, optional
            Whether the Poisson likelihood representing the total number of events is
            included in the analysis. Default value: True.
        
        luminosity : float, optional
            Integrated luminosity in pb^{-1} assumed in the analysis. Default value: 300000.
        
        n_asimov : int or None, optional
            Size of the Asimov sample. If None, all weighted events in the MadMiner
            file are used. Default value: None.
        
        mode : {"weighted" , "sampled"} , optional
            If "sampled", for each evaulation of the likelihood function, a separate
            set of events are sampled and histogram is created to construct the
            likelihood function. If "weighted", first a set of weighted events is
            sampled which is then used to create histograms. Default value: "sampled"
        
        n_histo_toys : int or None, optional
            Number of events drawn to construct the histograms used. If None and weighted_histo
            is True, all events in the training fraction of the MadMiner file are used. If None
            and weighted_histo is False, 100000 events are used. Default value: 100000.
        
        model_file : str or None, optional
            Filename of a saved neural network estimating the score. Required if
            score_components is not None. Default value: None.
        
        hist_bins : int or list of (int or ndarray) or None, optional
            Defines the histogram binning. If int, gives the number of bins automatically
            chosen for each summary statistic. If list, each entry corresponds to one
            summary statistic (e.g. kinematic variable specified by hist_vars); an int
            entry corresponds to the number of automatically chosen bins, an ndarray specifies
            the bin edges along this dimension explicitly. If None, the bins are chosen according
            to the defaults: for one summary statistic the default is 25 bins, for 2 it's 8 bins
            along each direction, for more it's 5 per dimension. Default value: None.
        
        thetas_binning : list of ndarray or None
            Specifies the parameter points used to determine the optimal binning.
            If none, theta_true will be used. Default value : None
            
        test_split :
        
        Returns
        -------
        negative_log_likelihood : likelihood
            Function that evaluates the negative log likelihood for a given parameter point
        
        """

        if thetas_binning is None:
            thetas_binning = [theta_true]

        x_asimov, x_weights = self._asimov_data(theta_true, n_asimov=n_asimov)
        n_observed = luminosity * self.xsecs([theta_true], [nu_true])[0]

        return self.create_negative_log_likelihood(
            observables=observables,
            score_components=score_components,
            x_observed=x_asimov,
            n_observed=n_observed,
            x_observed_weights=x_weights,
            include_xsec=include_xsec,
            luminosity=luminosity,
            mode=mode,
            n_histo_toys=n_histo_toys,
            model_file=model_file,
            hist_bins=hist_bins,
            thetas_binning=thetas_binning,
        )

    def _log_likelihood(
        self,
        n_events,
        xs,
        theta,
        nu,
        include_xsec=True,
        luminosity=300000.0,
        x_weights=None,
        mode="sampled",
        n_histo_toys=100000,
        hist_bins=None,
        summary_function=None,
        data=None,
        summary_stats=None,
        weights_benchmarks=None,
        benchmark_histograms=None,
        total_weights=None,
    ):
        """
        Low-level function which calculates the value of the log-likelihood ratio.
        See create_negative_log_likelihood for options.
        """

        log_likelihood = 0.0
        if include_xsec:
            log_likelihood = log_likelihood + self._log_likelihood_poisson(
                n_events, theta, nu, luminosity, weights_benchmarks, total_weights
            )

        if summary_function is not None:
            if x_weights is None:
                x_weights = n_events / float(len(xs)) * np.ones(len(xs))
            else:
                x_weights = x_weights * n_events / np.sum(x_weights)
            log_likelihood_events = self._log_likelihood_kinematic(
                xs,
                theta,
                nu,
                mode,
                n_histo_toys,
                hist_bins,
                summary_function,
                data,
                summary_stats,
                weights_benchmarks,
                benchmark_histograms,
            )
            log_likelihood = log_likelihood + np.dot(x_weights, log_likelihood_events)

        if nu is not None:
            log_likelihood = log_likelihood + self._log_likelihood_constraint(nu)

        logger.debug("Total log likelihood: %s", log_likelihood)
        return log_likelihood

    def _log_likelihood_kinematic(
        self,
        xs,
        theta,
        nu,
        mode="sampled",
        n_histo_toys=100000,
        hist_bins=None,
        summary_function=None,
        data=None,
        summary_stats=None,
        weights_benchmarks=None,
        benchmark_histograms=None,
    ):
        """
        Low-level function which calculates the value of the kinematic part of the
        log-likelihood. See create_negative_log_likelihood for options.
        """
        # shape of theta
        if nu is not None:
            theta = np.concatenate((theta, nu), axis=0)

        # Calculate summary statistics
        if summary_stats is None:
            summary_stats = summary_function(xs)

        # Make histograms
        if mode == "sampled":
            data = self._make_histo_data_sampled(
                summary_function=summary_function, theta=theta, n_histo_toys=n_histo_toys
            )
            histo = Histo(data, weights=None, bins=hist_bins, epsilon=1.0e-12)
        elif mode == "weighted":
            weights = self._weights([theta], [nu], weights_benchmarks)[0]
            histo = Histo(data, weights=weights, bins=hist_bins, epsilon=1.0e-12)
        elif mode == "histo":
            bin_centers = [np.array([(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]) for bins in hist_bins]
            bin_centers = np.array(list(product(*bin_centers)))
            histo = self._histogram_morphing(theta, benchmark_histograms, hist_bins, bin_centers)

        # calculate log-likelihood from histogram
        log_p = histo.log_likelihood(summary_stats)

        return log_p

    def _make_summary_statistic_function(self, observables=None, model=None):
        """
        Low-level function that returns a function "summary_function" which
        evaluates the summary statistic for an event.
        """
        variables = math_commands()
        x_indices = self._find_x_indices(observables)

        def summary_function(xs):
            # only prefined observables - very fast
            if not "score" in x_indices and not "function" in x_indices:
                return xs[:, x_indices]

            # evaulate some observables using eval() - more slow
            data_events = []
            for x in xs:
                data_event = []
                if "function" in x_indices:
                    for observable_name, observable_value in zip(self.observables, x):
                        variables[observable_name] = observable_value
                if "score" in x_indices:
                    if isinstance(model, ScoreEstimator):
                        score = model.evaluate_score(x=np.array([x]))[0]
                    elif isinstance(model, Ensemble) and model.estimator_type == "score":
                        score, _ = model.evaluate_score(x=np.array([x]), calculate_covariance=False)[0]
                    else:
                        raise RuntimeError("Model has to be 'ScoreEstimator' or Ensemble thereof.")

                for observable, x_indice in zip(observables, x_indices):
                    if x_indice == "function":
                        data_event.append(float(eval(observable, variables)))
                    elif x_indice == "score":
                        data_event.append(score[observable])
                    else:
                        data_event.append(x[x_indice])
                data_events.append(data_event)
            return np.array(data_events)

        return summary_function

    def _find_x_indices(self, observables):
        """
        Low-level function that finds the indices corresponding to the observables
        and returns them as a list.
        """
        x_names = list(self.observables.keys())
        x_indices = []
        for obs in observables:
            if isinstance(obs, int):
                x_indices.append("score")
            else:
                try:
                    x_indices.append(x_names.index(obs))
                except:
                    x_indices.append("function")
        logger.debug("Using x indices %s", x_indices)
        return x_indices

    def _make_histo_data_sampled(self, summary_function, theta, n_histo_toys=1000):
        """
        Low-level function that creates histogram data sampled from one benchmark
        """
        # Get unweighted events
        with less_logging():
            sampler = SampleAugmenter(self.madminer_filename, include_nuisance_parameters=True)
            x, theta, _ = sampler.sample_train_plain(
                theta=sampling.morphing_point(theta),
                n_samples=n_histo_toys,
                test_split=False,
                filename=None,
                folder=None,
            )

        # Calculate summary stats
        data = summary_function(x)

        return data

    def _make_histo_data_weighted(self, summary_function, n_toys, test_split=None):
        """
        Low-level function that creates weighted histogram data
        """
        # Get weighted events
        start_event, end_event, correction_factor = self._train_test_split(True, test_split)
        x, weights_benchmarks = self.weighted_events(start_event=start_event, end_event=end_event, n_draws=n_toys)
        weights_benchmarks *= self.n_samples / n_toys

        # Calculate summary stats
        if summary_function is not None:
            data = summary_function(x)
        else:
            data = None

        return data, weights_benchmarks

    def _find_bins(self, hist_bins, n_summary_stats):
        """
        Low-level function that sets up the binning of the histograms (I)
        """
        if hist_bins is None:
            if n_summary_stats == 1:
                hist_bins = [25]
            elif n_summary_stats == 2:
                hist_bins = [8, 8]
            else:
                hist_bins = [5 for _ in range(n_summary_stats)]
        elif isinstance(hist_bins, int):
            # hist_bins = tuple([hist_bins] * n_summary_stats)
            hist_bins = [hist_bins for _ in range(n_summary_stats)]
        return hist_bins

    def _fixed_adaptive_binning(
        self,
        thetas_binning,
        x_bins,
        data=None,
        weights_benchmarks=None,
        n_toys=None,
        test_split=None,
        summary_function=None,
    ):
        """
        Low-level function that sets up the binning of the histograms (II)
        """

        # Get weighted data
        if data is None:
            data, weights_benchmarks = self._make_histo_data_weighted(
                summary_function=summary_function, n_toys=n_toys, test_split=test_split
            )

        # Calculate weights for thetas
        weights = self._weights(thetas_binning, None, weights_benchmarks)
        weights = np.asarray(weights)
        weights = np.mean(weights, axis=0)

        # Histogram
        histo = Histo(data, weights, x_bins, epsilon=1.0e-12)
        x_bins = histo.edges
        return x_bins

    def _get_benchmark_histograms(self, data, weights_benchmarks, hist_bins, epsilon=1.0e-12):
        """
        Low-level function that returns histograms for morphing benchmarks
        """
        # get histogram bins
        histo_benchmarks = []
        for weights in weights_benchmarks.T:
            # histo = Histo(data, weights=weights, bins=hist_bins, epsilon=1.0e-12)
            ranges = [(edges[0], edges[-1]) for edges in hist_bins]
            histo, _ = np.histogramdd(data, bins=hist_bins, range=ranges, normed=False, weights=weights)
            histo[:] += epsilon
            histo_benchmarks.append(histo)

        # get bin centers
        bin_centers = [np.array([(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]) for bins in hist_bins]
        bin_centers = np.array(list(product(*bin_centers)))
        return histo_benchmarks, bin_centers

    def _histogram_morphing(self, theta, histogram_benchmarks, hist_bins, bin_centers):
        """
        Low-level function that morphes histograms
        """
        # get binning
        hist_nbins = [len(bins) - 1 for bins in hist_bins]

        # get array of flattened histograms
        flattened_histo_weights = []
        for histo in histogram_benchmarks:
            flattened_histo_weights.append(histo.flatten())
        flattened_histo_weights = np.array(flattened_histo_weights).T

        # calculate dot product
        theta_matrix = self._get_theta_benchmark_matrix(theta)
        histo_weights_theta = mdot(theta_matrix, flattened_histo_weights)

        # create histogram
        histo = Histo(bin_centers, weights=histo_weights_theta, bins=hist_bins, epsilon=1.0e-12)
        return histo

    def _clean_nans(self, array):
        not_finite = np.any(~np.isfinite(array), axis=0)
        if np.sum(not_finite) > 0:
            logger.warning("Removing %s inf / nan results from calculation")
            array[:, not_finite] = 0.0
        return array


def fix_params(negative_log_likelihood, theta, fixed_components):
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
        
    fixed_components : list of int
        m-dimensional vector of coordinate indices provided in theta.
        Example: fixed_components=[0,1] will fix the 1st and 2nd
        component of the paramater point.

    Returns
    -------
    constrained_nll_negative_log_likelihood : likelihood
        Constrained likelihood function which takes an
        n-m dimensional input parameter.
        
    """

    def constrained_nll(params):

        # Just return the expected Length
        n_dimension = negative_log_likelihood(None)
        if params is None:
            return n_dimension - len(fixed_components)

        # Process input
        if len(theta) != len(fixed_components):
            logger.warning("Length of fixed_components and theta should be the same")
        if len(params) + len(fixed_components) != n_dimension:
            logger.warning("Length of params should be %s", n_dimension - len(fixed_components))

        # Initialize full paramaters
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
    while the remaining N-M paramters are set to zero.
        
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
    while the remaining N-M paramters are profiled over.
        
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
        Mimization method used. Default value: "TNC"
        
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

    # Method
    supported_methods = ["TNC", " L-BFGS-B"]
    if method not in supported_methods:
        raise ValueError("Method %s unknown. Choose one of the following methods: %s", method, supported_methods)

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
