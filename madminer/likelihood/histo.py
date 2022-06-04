import logging
import numpy as np
from itertools import product

from .base import BaseLikelihood
from .. import sampling
from ..ml import ScoreEstimator, Ensemble, load_estimator
from ..utils.histo import Histo
from ..utils.various import mdot, less_logging, math_commands
from ..sampling import SampleAugmenter

logger = logging.getLogger(__name__)


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
        parameter point, evaluated with a dataset (x_observed,n_observed,x_observed_weights).

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
            If "sampled", for each evaluation of the likelihood function, a separate
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
            observables = []
        if score_components is None:
            score_components = []
        if observables == [] and score_components == []:
            logger.info("No observables and scores provided. Calculate LLR due to rate and set include_xsec=True.")
            include_xsec = True
        observables = list(observables) + list(score_components)

        if n_observed is None:
            n_observed = len(x_observed)

        if mode not in {"sampled", "weighted", "histo"}:
            raise ValueError(f"Mode {mode} unknown.")
        if mode == "histo" and self.n_nuisance_parameters > 0:
            raise ValueError("Mode histo is currently not supported in the presence of nuisance parameters")

        # Load model - nothing interesting
        if score_components:
            assert all(isinstance(score_component, int) for score_component in score_components)
            if model_file is None:
                raise ValueError("You need to provide a model_file!")
            model = load_estimator(model_file)
        else:
            model = None

        # Create summary function
        logger.info("Setting up standard summary statistics")
        summary_function = None
        if observables:
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
            hist_bins is None or not all(hasattr(hist_bin, "__len__") for hist_bin in hist_bins)
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
                    "Number of parameters is %s, expected %s physical parameters and %s nuisance parameters",
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
        parameter point, evaluated with test data sampled according to theta_true.

        Parameters
        ----------
        theta_true : ndarray
            Specifies the physical parameters according to which the test data is sampled.

        nu_true : ndarray
            Specifies the nuisance parameters according to which the test data is sampled.

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
            If "sampled", for each evaluation of the likelihood function, a separate
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
            bin_centers = [np.array([(a + b) / 2 for a, b in zip(bins[0:], bins[1:])]) for bins in hist_bins]
            bin_centers = np.array(list(product(*bin_centers)))
            histo = self._histogram_morphing(theta, benchmark_histograms, hist_bins, bin_centers)

        # calculate log-likelihood from histogram
        return histo.log_likelihood(summary_stats)

    def _make_summary_statistic_function(self, observables=None, model=None):
        """
        Low-level function that returns a function "summary_function" which
        evaluates the summary statistic for an event.
        """
        variables = math_commands()
        x_indices = self._find_x_indices(observables)

        def summary_function(xs):
            # only prefined observables - very fast
            if "score" not in x_indices and "function" not in x_indices:
                return xs[:, x_indices]

            # evaluate some observables using eval() - more slow
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

                for observable, x_index in zip(observables, x_indices):
                    if x_index == "function":
                        data_event.append(float(eval(observable, variables)))
                    elif x_index == "score":
                        data_event.append(score[observable])
                    else:
                        data_event.append(x[x_index])
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
        return summary_function(x)

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
                summary_function=summary_function,
                n_toys=n_toys,
                test_split=test_split,
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
        bin_centers = [np.array([(a + b) / 2 for a, b in zip(bins[0:], bins[1:])]) for bins in hist_bins]
        bin_centers = np.array(list(product(*bin_centers)))
        return histo_benchmarks, bin_centers

    def _histogram_morphing(self, theta, histogram_benchmarks, hist_bins, bin_centers):
        """
        Low-level function that morphes histograms
        """

        # get array of flattened histograms
        flattened_histo_weights = [histo.flatten() for histo in histogram_benchmarks]
        flattened_histo_weights = np.array(flattened_histo_weights).T

        # calculate dot product
        theta_matrix = self._get_theta_benchmark_matrix(theta)
        histo_weights_theta = mdot(theta_matrix, flattened_histo_weights)

        # create histogram
        return Histo(bin_centers, weights=histo_weights_theta, bins=hist_bins, epsilon=1.0e-12)
