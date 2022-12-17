import logging

from pathlib import Path

import numpy as np

from madminer.analysis import DataAnalyzer
from madminer.utils.various import math_commands
from madminer.utils.various import weighted_quantile
from madminer.utils.various import sanitize_array
from madminer.utils.various import mdot
from madminer.utils.various import less_logging
from madminer.ml import ParameterizedRatioEstimator
from madminer.ml import ScoreEstimator
from madminer.ml import Ensemble
from madminer.ml import load_estimator

logger = logging.getLogger(__name__)


class FisherInformation(DataAnalyzer):
    """
    Functions to calculate expected Fisher information matrices.

    After initializing a `FisherInformation` instance with the filename of a MadMiner file, different information matrices
    can be calculated:

    * `FisherInformation.truth_information()` calculates the full truth-level Fisher information.
      This is the information in an idealized measurement where all parton-level particles with their charges, flavours,
      and four-momenta can be accessed with perfect accuracy.
    * `FisherInformation.full_information()` calculates the full Fisher information in
      realistic detector-level observations, estimated with neural networks. In addition to the MadMiner file, this
      requires a trained SALLY or SALLINO estimator as well as an unweighted evaluation sample.
    * `FisherInformation.rate_information()` calculates the Fisher information in the total cross
      section.
    * `FisherInformation.histo_information()` calculates the Fisher information in the histogram of
      one (parton-level or detector-level) observable.
    * `FisherInformation.histo_information_2d()` calculates the Fisher information in a two-dimensional
      histogram of two (parton-level or detector-level) observables.
    * `FisherInformation.histogram_of_information()` calculates the full truth-level Fisher information in
      different slices of one observable (the "distribution of the Fisher information").

    Finally, don't forget that in the presence of nuisance parameters the constraint terms also affect the Fisher
    information. This term is given by `FisherInformation.calculate_fisher_information_nuisance_constraints()`.

    Parameters
    ----------
    filename : str
        Path to MadMiner file (for instance the output of `madminer.delphes.DelphesProcessor.save()`).

    include_nuisance_parameters : bool, optional
        If True, nuisance parameters are taken into account. Default value: True.

    """

    def __init__(self, filename, include_nuisance_parameters=True):
        super().__init__(filename, False, include_nuisance_parameters)

    def truth_information(
        self,
        theta,
        luminosity=300000.0,
        cuts=None,
        efficiency_functions=None,
        include_nuisance_parameters=True,
    ):
        """
        Calculates the full Fisher information at parton / truth level. This is the information in an idealized
        measurement where all parton-level particles with their charges, flavours, and four-momenta can be accessed with
        perfect accuracy, i.e. the latent variables `z_parton` can be measured directly.

        Parameters
        ----------
        theta : ndarray
            Parameter point `theta` at which the Fisher information matrix `I_ij(theta)` is evaluated.

        luminosity : float
            Luminosity in pb^-1.

        cuts : None or list of str, optional
            Cuts. Each entry is a parseable Python expression that returns a bool (True if the event should pass a cut,
            False otherwise). Default value: None.

        efficiency_functions : list of str or None
            Efficiencies. Each entry is a parseable Python expression that returns a float for the efficiency of one
            component. Default value: None.

        include_nuisance_parameters : bool, optional
            If True, nuisance parameters are taken into account. Default value: True.

        Returns
        -------
        fisher_information : ndarray
            Expected full truth-level Fisher information matrix with shape `(n_parameters, n_parameters)`.

        fisher_information_uncertainty : ndarray
            Covariance matrix of the Fisher information matrix with shape
            `(n_parameters, n_parameters, n_parameters, n_parameters)`, calculated with plain Gaussian error
            propagation.

        """

        # Input
        if cuts is None:
            cuts = []
        if efficiency_functions is None:
            efficiency_functions = []

        include_nuisance_parameters = include_nuisance_parameters and self.n_nuisance_parameters > 0

        # Loop over batches
        n_all_parameters = self.n_parameters
        if include_nuisance_parameters:
            n_all_parameters += self.n_nuisance_parameters

        fisher_info = np.zeros((n_all_parameters, n_all_parameters))
        covariance = np.zeros((n_all_parameters, n_all_parameters, n_all_parameters, n_all_parameters))

        for observations, weights in self.event_loader():
            # Cuts
            cut_filter = [self._pass_cuts(obs_event, cuts) for obs_event in observations]
            observations = observations[cut_filter]
            weights = weights[cut_filter]

            # Efficiencies
            efficiencies = np.array(
                [self._eval_efficiency(obs_event, efficiency_functions) for obs_event in observations]
            )
            weights *= efficiencies[:, np.newaxis]

            # Fisher information
            this_fisher_info, this_covariance = self._calculate_fisher_information(
                theta,
                weights,
                luminosity,
                sum_events=True,
                calculate_uncertainty=True,
                include_nuisance_parameters=include_nuisance_parameters,
            )
            fisher_info += this_fisher_info
            covariance += this_covariance

        return fisher_info, covariance

    def full_information(
        self,
        theta,
        model_file,
        unweighted_x_sample_file=None,
        luminosity=300000.0,
        include_xsec_info=True,
        mode="score",
        calculate_covariance=True,
        batch_size=100000,
        test_split=0.2,
    ):
        """
        Calculates the full Fisher information in realistic detector-level observations, estimated with neural networks.
        In addition to the MadMiner file, this requires a trained SALLY or SALLINO estimator.

        Nuisance parameter are taken into account automatically if the SALLY / SALLINO model was trained with them.

        Parameters
        ----------
        theta : ndarray
            Parameter point `theta` at which the Fisher information matrix `I_ij(theta)` is evaluated.

        model_file : str
            Filename of a trained local score regression model that was trained on samples from `theta` (see
            `madminer.ml.Estimator`).

        unweighted_x_sample_file : str or None
            Filename of an unweighted x sample that is sampled according to theta and obeys the cuts
            (see `madminer.sampling.SampleAugmenter.extract_samples_train_local()`). If None, the Fisher information
            is instead calculated on the full, weighted samples (the data in the MadMiner file). Default value: None.

        luminosity : float, optional
            Luminosity in pb^-1. Default value: 300000.

        include_xsec_info : bool, optional
            Whether the rate information is included in the returned Fisher information. Default value: True.

        mode : {"score", "information"}, optional
            How the ensemble uncertainty on the kinematic Fisher information is calculated. If mode is "information",
            the Fisher information for each estimator is calculated individually and only then
            are the sample mean and covariance calculated. If mode is "score", the sample mean is
            calculated for the score for each event. Default value: "score".

        calculate_covariance : bool, optional
            If True, the covariance between the different estimators is calculated. Default value: True.

        batch_size : int, optional
            Batch size. Default value: 100000.

        test_split : float or None, optional
            If unweighted_x_sample_file is None, this determines the fraction of weighted events used for evaluation.
            If None, all events are used (this will probably include events used during training!). Default value: 0.2.

        Returns
        -------
        fisher_information : ndarray or list of ndarray
            Estimated expected full detector-level Fisher information matrix with shape `(n_parameters, n_parameters)`.
            If more then one value ensemble_vote_expectation_weight is given, this is a list with results for all
            entries in ensemble_vote_expectation_weight.

        fisher_information_uncertainty : ndarray or list of ndarray or None
            Covariance matrix of the Fisher information matrix with shape
            `(n_parameters, n_parameters, n_parameters, n_parameters)`. If more then one value
            ensemble_vote_expectation_weight is given, this is a list with results for all entries in
            ensemble_vote_expectation_weight.
        """

        # Check input
        if mode not in {"score", "information", "modified_score"}:
            raise ValueError(f"Unknown mode {mode}")

        # Load Estimator model
        if Path(model_file).is_dir() and Path(model_file, "ensemble.json").exists():
            model_is_ensemble = True
            model = Ensemble()
            model.load(model_file)
            if isinstance(model.estimators[0], ParameterizedRatioEstimator):
                model_type = "Parameterized Ratio Ensemble"
            elif isinstance(model.estimators[0], ScoreEstimator):
                model_type = "Score Ensemble"
            else:
                raise RuntimeError("Ensemble is not a score or parameterized_ratio type!")
        else:
            model_is_ensemble = False
            model = load_estimator(model_file)

            if isinstance(model, ParameterizedRatioEstimator):
                model_type = "Parameterized Ratio Estimator"
            elif isinstance(model, ScoreEstimator):
                model_type = "Score Estimator"
            else:
                raise RuntimeError("Estimator is not a score or parameterized_ratio type!")

        # Nuisance parameters?
        if model.n_parameters == self.n_parameters:
            logger.info(
                "Found %s parameters in %s model, matching %s physical parameters in MadMiner file",
                model.n_parameters,
                model_type,
                self.n_parameters,
            )
            include_nuisance_parameters = False
        elif model.n_parameters == self.n_parameters + self.n_nuisance_parameters:
            logger.info(
                "Found %s parameters in %s model, "
                "matching %s physical parameters + %s nuisance parameters in MadMiner file",
                model.n_parameters,
                model_type,
                self.n_parameters,
                self.n_nuisance_parameters,
            )
            include_nuisance_parameters = True
        else:
            raise RuntimeError(
                f"Inconsistent numbers of parameters! "
                f"Found {model.n_parameters} in {model_type} model, "
                f"but {self.n_parameters} physical parameters, "
                f"and {self.n_nuisance_parameters} nuisance parameters in MadMiner file."
            )

        if include_nuisance_parameters:
            logger.debug("Including nuisance parameters")
        else:
            logger.debug("Not including nuisance parameters")

        # Total xsec
        total_xsec = self._calculate_xsec(theta=theta)
        logger.debug("Total cross section: %s pb", total_xsec)

        # Rate part of Fisher information
        fisher_info_rate = 0.0
        rate_covariance = 0.0
        if include_xsec_info:
            logger.info("Evaluating rate Fisher information")
            fisher_info_rate, rate_covariance = self.rate_information(
                theta=theta,
                luminosity=luminosity,
                include_nuisance_parameters=include_nuisance_parameters,
            )

        # Evaluation from weighted events
        if unweighted_x_sample_file is None:

            # Which events to sum over
            if test_split is None or test_split <= 0.0 or test_split >= 1.0:
                start_event = 0
            else:
                start_event = int(round((1.0 - test_split) * self.n_samples, 0)) + 1

            if start_event > 0:
                total_sum_weights_theta = self._calculate_xsec(theta=theta, start_event=start_event)
            else:
                total_sum_weights_theta = total_xsec

            # Theta morphing matrix
            theta_matrix = self._get_theta_benchmark_matrix(theta)

            # Prepare output
            fisher_info_kin = None
            covariance = None

            # Number of batches
            n_batches = int(np.ceil((self.n_samples - start_event) / batch_size))
            n_batches_verbose = max(int(round(n_batches / 10, 0)), 1)

            events = self.event_loader(
                batch_size=batch_size,
                start=start_event,
                include_nuisance_parameters=include_nuisance_parameters,
            )

            for i_batch, (observations, weights_benchmarks) in enumerate(events, start=1):
                if i_batch % n_batches_verbose == 0:
                    logger.info("Evaluating kinematic Fisher information on batch %s / %s", i_batch, n_batches)
                else:
                    logger.debug("Evaluating kinematic Fisher information on batch %s / %s", i_batch, n_batches)

                weights_theta = mdot(theta_matrix, weights_benchmarks)

                # Calculate Fisher info on this batch
                if model_is_ensemble:
                    with less_logging():
                        this_fisher_info, this_covariance = model.calculate_fisher_information(
                            x=observations,
                            theta=theta,
                            obs_weights=weights_theta,
                            n_events=luminosity * total_xsec * np.sum(weights_theta) / total_sum_weights_theta,
                            calculate_covariance=calculate_covariance,
                            mode=mode,
                        )
                else:
                    with less_logging():
                        this_fisher_info = model.calculate_fisher_information(
                            x=observations,
                            theta=theta,
                            weights=weights_theta,
                            n_events=luminosity * total_xsec * np.sum(weights_theta) / total_sum_weights_theta,
                        )
                    this_covariance = None

                # Sum up results
                if fisher_info_kin is None:
                    fisher_info_kin = this_fisher_info
                elif isinstance(fisher_info_kin, list):
                    for i in range(len(fisher_info_kin)):
                        fisher_info_kin[i] += this_fisher_info[i]
                else:
                    fisher_info_kin += this_fisher_info

                if this_covariance is not None:
                    if covariance is None:
                        covariance = this_covariance
                    elif isinstance(covariance, list):
                        for i in range(len(covariance)):
                            covariance[i] += this_covariance[i]
                    else:
                        covariance += this_covariance

        # Evaluation from unweighted event sample
        else:
            with less_logging():
                if model_is_ensemble:
                    fisher_info_kin, covariance = model.calculate_fisher_information(
                        x=unweighted_x_sample_file,
                        theta=theta,
                        n_events=luminosity * total_xsec,
                        mode=mode,
                        calculate_covariance=calculate_covariance,
                    )
                else:
                    fisher_info_kin = model.calculate_fisher_information(
                        x=unweighted_x_sample_file,
                        theta=theta,
                        n_events=luminosity * total_xsec,
                    )
                    covariance = None

        # Returns
        if model_is_ensemble and calculate_covariance:
            return fisher_info_rate + fisher_info_kin, rate_covariance + covariance
        else:
            return fisher_info_rate + fisher_info_kin, rate_covariance

    def rate_information(
        self,
        theta,
        luminosity,
        cuts=None,
        efficiency_functions=None,
        include_nuisance_parameters=True,
    ):
        """
        Calculates the Fisher information in a measurement of the total cross section (without any kinematic
        information).

        Parameters
        ----------
        theta : ndarray
            Parameter point `theta` at which the Fisher information matrix `I_ij(theta)` is evaluated.

        luminosity : float
            Luminosity in pb^-1.

        cuts : None or list of str, optional
            Cuts. Each entry is a parseable Python expression that returns a bool (True if the event should pass a cut,
            False otherwise). Default value: None.

        efficiency_functions : list of str or None
            Efficiencies. Each entry is a parseable Python expression that returns a float for the efficiency of one
            component. Default value: None.

        include_nuisance_parameters : bool, optional
            If True, nuisance parameters are taken into account. Default value: True.

        Returns
        -------
        fisher_information : ndarray
            Expected Fisher information in the total cross section with shape `(n_parameters, n_parameters)`.

        fisher_information_uncertainty : ndarray
            Covariance matrix of the Fisher information matrix with shape
            `(n_parameters, n_parameters, n_parameters, n_parameters)`, calculated with plain Gaussian error
            propagation.
        """

        include_nuisance_parameters = include_nuisance_parameters and self.n_nuisance_parameters > 0

        # Get weights at benchmarks
        weights_benchmarks, weights_benchmark_uncertainties = self._calculate_xsec(
            cuts=cuts,
            efficiency_functions=efficiency_functions,
            return_benchmark_xsecs=True,
            return_error=True,
            include_nuisance_parameters=include_nuisance_parameters,
        )

        weights_benchmarks = weights_benchmarks.reshape((1, -1))
        weights_benchmark_uncertainties = weights_benchmark_uncertainties.reshape((1, -1))

        # Get Fisher information
        fisher_info, covariance = self._calculate_fisher_information(
            theta=theta,
            weights_benchmarks=weights_benchmarks,
            luminosity=luminosity,
            sum_events=True,
            calculate_uncertainty=True,
            weights_benchmark_uncertainties=weights_benchmark_uncertainties,
            include_nuisance_parameters=include_nuisance_parameters,
        )

        return fisher_info, covariance

    def histo_information(
        self,
        theta,
        luminosity,
        observable,
        bins,
        histrange=None,
        cuts=None,
        efficiency_functions=None,
        n_events_dynamic_binning=None,
    ):
        """
        Calculates the Fisher information in the one-dimensional histogram of an (parton-level or detector-level,
        depending on how the observations in the MadMiner file were calculated) observable.

        Parameters
        ----------
        theta : ndarray
            Parameter point `theta` at which the Fisher information matrix `I_ij(theta)` is evaluated.

        luminosity : float
            Luminosity in pb^-1.

        observable : str
            Expression for the observable to be histogrammed. The str will be parsed by Python's `eval()` function
            and can use the names of the observables in the MadMiner files.

        bins : int or ndarray
            If int: number of bins in the histogram, excluding overflow bins. Otherwise, defines the bin boundaries
            (excluding overflow bins).

        histrange : tuple of float or None, optional
            Minimum and maximum value of the histogram in the form `(min, max)`. Overflow bins are always added. If
            None and bins is an int, variable-width bins with equal cross section are constructed automatically.
            Default value: None.

        cuts : None or list of str, optional
            Cuts. Each entry is a parseable Python expression that returns a bool (True if the event should pass a cut,
            False otherwise). Default value: None.

        efficiency_functions : list of str or None
            Efficiencies. Each entry is a parseable Python expression that returns a float for the efficiency of one
            component. Default value: None.

        n_events_dynamic_binning : int or None, optional
            Number of events used to calculate the dynamic binning (if histrange is None). If None, all events are used.
            Note that these events are not shuffled, so if the events in the MadMiner file are sorted, using a value
            different from None can cause issues. Default value: None.

        Returns
        -------
        fisher_information : ndarray
            Expected Fisher information in the histogram with shape `(n_parameters, n_parameters)`.

        fisher_information_uncertainty : ndarray
            Covariance matrix of the Fisher information matrix with shape
            `(n_parameters, n_parameters, n_parameters, n_parameters)`, calculated with plain Gaussian error
            propagation.

        """

        # Input
        if cuts is None:
            cuts = []
        if efficiency_functions is None:
            efficiency_functions = []

        # Binning
        bin_boundaries, n_bins_total = self._calculate_binning(
            bins, cuts, efficiency_functions, histrange, n_events_dynamic_binning, observable, theta
        )

        # Loop over batches
        weights_benchmarks = np.zeros((n_bins_total, self.n_benchmarks))
        weights_squared_benchmarks = np.zeros((n_bins_total, self.n_benchmarks))

        for observations, weights in self.event_loader():
            # Cuts
            cut_filter = [self._pass_cuts(obs_event, cuts) for obs_event in observations]
            observations = observations[cut_filter]
            weights = weights[cut_filter]

            # Efficiencies
            efficiencies = np.array(
                [self._eval_efficiency(obs_event, efficiency_functions) for obs_event in observations]
            )
            weights *= efficiencies[:, np.newaxis]

            # Evaluate histogrammed observable
            histo_observables = np.asarray([self._eval_observable(obs_event, observable) for obs_event in observations])

            # Find bins
            i_bins = np.searchsorted(bin_boundaries, histo_observables)
            assert ((0 <= i_bins) & (i_bins < n_bins_total)).all(), f"Wrong bin {i_bins}"

            # Add up
            for i in range(n_bins_total):
                if len(weights[i_bins == i]) > 0:
                    weights_benchmarks[i] += np.sum(weights[i_bins == i], axis=0)
                    weights_squared_benchmarks[i] += np.sum(weights[i_bins == i] ** 2, axis=0)

        weights_benchmark_uncertainties = weights_squared_benchmarks**0.5

        # Check cross sections per bin
        self._check_binning_stats(weights_benchmarks, weights_benchmark_uncertainties, theta)

        # Calculate Fisher information in histogram
        fisher_info, covariance = self._calculate_fisher_information(
            theta,
            weights_benchmarks,
            luminosity,
            sum_events=True,
            weights_benchmark_uncertainties=weights_benchmark_uncertainties,
            calculate_uncertainty=True,
        )
        return fisher_info, covariance

    def histo_information_2d(
        self,
        theta,
        luminosity,
        observable1,
        bins1,
        observable2,
        bins2,
        histrange1=None,
        histrange2=None,
        cuts=None,
        efficiency_functions=None,
        n_events_dynamic_binning=None,
    ):

        """
        Calculates the Fisher information in a two-dimensional histogram of two (parton-level or detector-level,
        depending on how the observations in the MadMiner file were calculated) observables.

        Parameters
        ----------
        theta : ndarray
            Parameter point `theta` at which the Fisher information matrix `I_ij(theta)` is evaluated.

        luminosity : float
            Luminosity in pb^-1.

        observable1 : str
            Expression for the first observable to be histogrammed. The str will be parsed by Python's `eval()` function
            and can use the names of the observables in the MadMiner files.

        bins1 : int or ndarray
            If int: number of bins along the first axis in the histogram in the histogram, excluding overflow bins.
            Otherwise, defines the bin boundaries along the first axis in the histogram (excluding overflow bins).

        observable2 : str
            Expression for the first observable to be histogrammed. The str will be parsed by Python's `eval()` function
            and can use the names of the observables in the MadMiner files.

        bins2 : int or ndarray
            If int: number of bins along the second axis in the histogram in the histogram, excluding overflow bins.
            Otherwise, defines the bin boundaries along the second axis in the histogram (excluding overflow bins).

        histrange1 : tuple of float or None, optional
            Minimum and maximum value of the first axis of the histogram in the form `(min, max)`. Overflow bins are
            always added. If None, variable-width bins with equal cross section are constructed automatically. Default
            value: None.

        histrange2 : tuple of float or None, optional
            Minimum and maximum value of the first axis of the histogram in the form `(min, max)`. Overflow bins are
            always added. If None, variable-width bins with equal cross section are constructed automatically. Default
            value: None.

        cuts : None or list of str, optional
            Cuts. Each entry is a parseable Python expression that returns a bool (True if the event should pass a cut,
            False otherwise). Default value: None.

        efficiency_functions : list of str or None
            Efficiencies. Each entry is a parseable Python expression that returns a float for the efficiency of one
            component. Default value: None.

        n_events_dynamic_binning : int or None, optional
            Number of events used to calculate the dynamic binning (if histrange is None). If None, all events are used.
            Note that these events are not shuffled, so if the events in the MadMiner file are sorted, using a value
            different from None can cause issues. Default value: None.

        Returns
        -------
        fisher_information : ndarray
            Expected Fisher information in the histogram with shape `(n_parameters, n_parameters)`.

        fisher_information_uncertainty : ndarray
            Covariance matrix of the Fisher information matrix with shape
            `(n_parameters, n_parameters, n_parameters, n_parameters)`, calculated with plain Gaussian error
            propagation.

        """

        # Input
        if cuts is None:
            cuts = []
        if efficiency_functions is None:
            efficiency_functions = []

        # Binning
        bin1_boundaries, n_bins1_total = self._calculate_binning(
            bins1,
            cuts,
            efficiency_functions,
            histrange1,
            n_events_dynamic_binning,
            observable1,
            theta,
        )

        bin2_boundaries, n_bins2_total = self._calculate_binning(
            bins2,
            cuts,
            efficiency_functions,
            histrange2,
            n_events_dynamic_binning,
            observable2,
            theta,
        )

        # Loop over batches
        weights_benchmarks = np.zeros((n_bins1_total, n_bins2_total, self.n_benchmarks))
        weights_squared_benchmarks = np.zeros((n_bins1_total, n_bins2_total, self.n_benchmarks))

        for observations, weights in self.event_loader():
            # Cuts
            cut_filter = [self._pass_cuts(obs_event, cuts) for obs_event in observations]
            observations = observations[cut_filter]
            weights = weights[cut_filter]

            # Efficiencies
            efficiencies = np.array(
                [self._eval_efficiency(obs_event, efficiency_functions) for obs_event in observations]
            )
            weights *= efficiencies[:, np.newaxis]

            # Evaluate histogrammed observable
            histo1_observables = np.asarray(
                [self._eval_observable(obs_event, observable1) for obs_event in observations]
            )
            histo2_observables = np.asarray(
                [self._eval_observable(obs_event, observable2) for obs_event in observations]
            )

            # Find bins
            i_bins1 = np.searchsorted(bin1_boundaries, histo1_observables)
            i_bins2 = np.searchsorted(bin2_boundaries, histo2_observables)

            assert ((0 <= i_bins1) & (i_bins1 < n_bins1_total)).all(), f"Wrong bin {i_bins1}"
            assert ((0 <= i_bins2) & (i_bins2 < n_bins1_total)).all(), f"Wrong bin {i_bins2}"

            # Add up
            for i in range(n_bins1_total):
                for j in range(n_bins2_total):
                    if len(weights[(i_bins1 == i) & (i_bins2 == j)]) > 0:
                        weights_benchmarks[i, j] += np.sum(weights[(i_bins1 == i) & (i_bins2 == j)], axis=0)
                        weights_squared_benchmarks[i, j] += np.sum(
                            weights[(i_bins1 == i) & (i_bins2 == j)] ** 2, axis=0
                        )

        weights_benchmark_uncertainties = weights_squared_benchmarks**0.5

        # Calculate Fisher information in histogram
        weights_benchmarks = weights_benchmarks.reshape(-1, self.n_benchmarks)
        weights_benchmark_uncertainties = weights_benchmark_uncertainties.reshape(-1, self.n_benchmarks)

        self._check_binning_stats(
            weights_benchmarks, weights_benchmark_uncertainties, theta, n_bins_last_axis=n_bins2_total
        )

        fisher_info, covariance = self._calculate_fisher_information(
            theta,
            weights_benchmarks,
            luminosity,
            sum_events=True,
            weights_benchmark_uncertainties=weights_benchmark_uncertainties,
            calculate_uncertainty=True,
        )

        return fisher_info, covariance

    def histogram_of_information(
        self,
        theta,
        observable,
        nbins,
        histrange,
        model_file=None,
        luminosity=300000.0,
        cuts=None,
        efficiency_functions=None,
        batch_size=100000,
        test_split=0.2,
    ):
        """
        Calculates the full and rate-only Fisher information in slices of one observable. For the full
        information, it will return the truth-level information if model_file is None, and otherwise the
        detector-level information based on the SALLY-type score estimator saved in model_file.

        Parameters
        ----------
        theta : ndarray
            Parameter point `theta` at which the Fisher information matrix `I_ij(theta)` is evaluated.

        observable : str
            Expression for the observable to be sliced. The str will be parsed by Python's `eval()` function
            and can use the names of the observables in the MadMiner files.

        nbins : int
            Number of bins in the slicing, excluding overflow bins.

        histrange : tuple of float
            Minimum and maximum value of the slicing in the form `(min, max)`. Overflow bins are always added.

        model_file : str or None, optional
            If None, the truth-level Fisher information is calculated. If str, filename of a trained local score
            regression model that was trained on samples from `theta` (see `madminer.ml.Estimator`). Default value:
            None.

        luminosity : float, optional
            Luminosity in pb^-1. Default value: 300000.

        cuts : None or list of str, optional
            Cuts. Each entry is a parseable Python expression that returns a bool (True if the event should pass a cut,
            False otherwise). Default value: None.

        efficiency_functions : list of str or None
            Efficiencies. Each entry is a parseable Python expression that returns a float for the efficiency of one
            component. Default value: None.

        batch_size : int, optional
            If model_file is not None: Batch size. Default value: 100000.

        test_split : float or None, optional
            If model_file is not None: If unweighted_x_sample_file is None, this determines the fraction of weighted
            events used for evaluation.
            If None, all events are used (this will probably include events used during training!). Default value: 0.2.


        Returns
        -------
        bin_boundaries : ndarray
            Observable slice boundaries.

        sigma_bins : ndarray
            Cross section in pb in each of the slices.

        fisher_infos_rate : ndarray
            Expected rate-only Fisher information for each slice. Has shape `(n_slices, n_parameters, n_parameters)`.

        fisher_infos_full : ndarray
            Expected full Fisher information for each slice. Has shape
            `(n_slices, n_parameters, n_parameters)`.

        """

        # Input
        if cuts is None:
            cuts = []
        if efficiency_functions is None:
            efficiency_functions = []

        # Theta morphing matrix
        theta_matrix = self._get_theta_benchmark_matrix(theta)

        # Number of bins
        n_bins_total = nbins + 2
        bin_boundaries = np.linspace(histrange[0], histrange[1], num=nbins + 1)

        # Prepare output
        weights_benchmarks_bins = np.zeros((n_bins_total, self.n_benchmarks))
        fisher_info_full_bins = np.zeros((n_bins_total, self.n_parameters, self.n_parameters))

        # Main loop: truth-level case
        if model_file is None:
            for observations, weights in self.event_loader():
                # Cuts
                cut_filter = [self._pass_cuts(obs_event, cuts) for obs_event in observations]
                observations = observations[cut_filter]
                weights = weights[cut_filter]

                # Efficiencies
                efficiencies = np.array(
                    [self._eval_efficiency(obs_event, efficiency_functions) for obs_event in observations]
                )
                weights *= efficiencies[:, np.newaxis]

                # Fisher info per event
                fisher_info_events = self._calculate_fisher_information(theta, weights, luminosity, sum_events=False)

                # Evaluate histogrammed observable
                histo_observables = np.asarray(
                    [self._eval_observable(obs_event, observable) for obs_event in observations]
                )

                # Get rid of nuisance parameters
                fisher_info_events = fisher_info_events[:, : self.n_parameters, : self.n_parameters]

                # Find bins
                bins = np.searchsorted(bin_boundaries, histo_observables)
                assert ((0 <= bins) & (bins < n_bins_total)).all(), f"Wrong bin {bins}"

                # Add up
                for i in range(n_bins_total):
                    if len(weights[bins == i]) > 0:
                        weights_benchmarks_bins[i] += np.sum(weights[bins == i], axis=0)
                        fisher_info_full_bins[i] += np.sum(fisher_info_events[bins == i], axis=0)

        # ML case
        else:
            # Load SALLY model
            if Path(model_file).is_dir() and Path(model_file, "ensemble.json").exists():
                model_is_ensemble = True
                model = Ensemble()
                model.load(model_file)
            else:
                model_is_ensemble = False
                model = ScoreEstimator()
                model.load(model_file)

            # Nuisance parameters?
            if model.n_parameters == self.n_parameters:
                logger.debug(
                    "Found %s parameters in SALLY model, matching %s physical parameters in MadMiner file",
                    model.n_parameters,
                    self.n_parameters,
                )
                include_nuisance_parameters = False
            elif model.n_parameters == self.n_parameters + self.n_nuisance_parameters:
                logger.debug(
                    "Found %s parameters in SALLY model, matching %s physical parameters + %s nuisance parameters"
                    + " in MadMiner file",
                    model.n_parameters,
                    self.n_parameters,
                    self.n_nuisance_parameters,
                )
                include_nuisance_parameters = True
            else:
                raise RuntimeError(
                    f"Inconsistent numbers of parameters! "
                    f"Found {model.n_parameters} in SALLY model, "
                    f"but {self.n_parameters} physical parameters in MadMiner file, "
                    f"and {self.n_nuisance_parameters} nuisance parameters in MadMiner file."
                )

            # Total xsec
            total_xsec = self._calculate_xsec(theta=theta)
            logger.debug("Total cross section: %s pb", total_xsec)

            # Which events to sum over
            if test_split is None or test_split <= 0.0 or test_split >= 1.0:
                start_event = 0
            else:
                start_event = int(round((1.0 - test_split) * self.n_samples, 0)) + 1

            # Number of batches
            n_batches = int(np.ceil((self.n_samples - start_event) / batch_size))
            n_batches_verbose = max(int(round(n_batches / 10, 0)), 1)

            events = self.event_loader(
                batch_size=batch_size,
                start=start_event,
                include_nuisance_parameters=include_nuisance_parameters,
            )

            # ML main loop
            for i_batch, (observations, weights_benchmarks) in enumerate(events, start=1):
                if i_batch % n_batches_verbose == 0:
                    logger.info("Evaluating kinematic Fisher information on batch %s / %s", i_batch, n_batches)
                else:
                    logger.debug("Evaluating kinematic Fisher information on batch %s / %s", i_batch, n_batches)

                # Cuts
                cut_filter = [self._pass_cuts(obs_event, cuts) for obs_event in observations]
                observations = observations[cut_filter]
                weights_benchmarks = weights_benchmarks[cut_filter]

                # Efficiencies
                efficiencies = np.array(
                    [self._eval_efficiency(obs_event, efficiency_functions) for obs_event in observations]
                )
                weights_benchmarks *= efficiencies[:, np.newaxis]

                # Rescale for test_split
                if test_split is not None:
                    correction = np.array([1.0 / test_split for _ in observations])
                    weights_benchmarks *= correction[:, np.newaxis]

                weights_theta = mdot(theta_matrix, weights_benchmarks)

                # Calculate Fisher info on this batch
                if model_is_ensemble:
                    fisher_info_events, _ = model.calculate_fisher_information(
                        x=observations,
                        obs_weights=weights_theta,
                        n_events=luminosity * np.sum(weights_theta),
                        mode="score",
                        calculate_covariance=False,
                        sum_events=False,
                    )
                else:
                    fisher_info_events = model.calculate_fisher_information(
                        x=observations,
                        weights=weights_theta,
                        n_events=luminosity * np.sum(weights_theta),
                        sum_events=False,
                    )

                # Get rid of nuisance parameters
                if include_nuisance_parameters:
                    fisher_info_events = fisher_info_events[:, : self.n_parameters, : self.n_parameters]

                # Evaluate histogrammed observable
                histo_observables = np.asarray(
                    [self._eval_observable(obs_event, observable) for obs_event in observations]
                )

                # Find bins
                bins = np.searchsorted(bin_boundaries, histo_observables)
                assert ((0 <= bins) & (bins < n_bins_total)).all(), f"Wrong bin {bins}"

                # Add up
                for i in range(n_bins_total):
                    if len(weights_benchmarks[bins == i]) > 0:
                        weights_benchmarks_bins[i] += np.sum(weights_benchmarks[bins == i], axis=0)
                        fisher_info_full_bins[i] += np.sum(fisher_info_events[bins == i], axis=0)

        # Calculate xsecs in bins
        sigma_bins = mdot(theta_matrix, weights_benchmarks_bins)  # (n_bins,)

        # Calculate rate-only Fisher informations in bins
        fisher_info_rate_bins = self._calculate_fisher_information(
            theta, weights_benchmarks_bins, luminosity, sum_events=False
        )

        # Get rid of nuisance parameters
        fisher_info_rate_bins = fisher_info_rate_bins[:, : self.n_parameters, : self.n_parameters]

        # If ML: xsec info is still missing !
        if model_file is not None:
            fisher_info_full_bins += fisher_info_rate_bins

        return bin_boundaries, sigma_bins, fisher_info_rate_bins, fisher_info_full_bins

    def histogram_of_sigma_dsigma(self, theta, observable, nbins, histrange, cuts=None, efficiency_functions=None):
        """
        Fills events into histograms and calculates the cross section and first derivative for each bin

        Parameters
        ----------
        theta : ndarray
            Parameter point `theta` at which the Fisher information matrix `I_ij(theta)` is evaluated.

        observable : str
            Expression for the observable to be sliced. The str will be parsed by Python's `eval()` function
            and can use the names of the observables in the MadMiner files.

        nbins : int
            Number of bins in the slicing, excluding overflow bins.

        histrange : tuple of float
            Minimum and maximum value of the slicing in the form `(min, max)`. Overflow bins are always added.

        cuts : None or list of str, optional
            Cuts. Each entry is a parseable Python expression that returns a bool (True if the event should pass a cut,
            False otherwise). Default value: None.

        efficiency_functions : list of str or None
            Efficiencies. Each entry is a parseable Python expression that returns a float for the efficiency of one
            component. Default value: None.

        Returns
        -------
        bin_boundaries : ndarray
            Observable slice boundaries.

        sigma_bins : ndarray
            Cross section in pb in each of the slices.

        dsigma_bins : ndarray
            Cross section in pb in each of the slices.
        """

        # Input
        if cuts is None:
            cuts = []
        if efficiency_functions is None:
            efficiency_functions = []

        # Binning
        dynamic_binning = histrange is None
        if dynamic_binning:
            n_bins_total = nbins
            bin_boundaries = self._calculate_dynamic_binning(observable, theta, nbins, None, cuts, efficiency_functions)
        else:
            n_bins_total = nbins + 2
            bin_boundaries = np.linspace(histrange[0], histrange[1], num=nbins + 1)

        #        # Number of bins
        # n_bins_total = nbins + 2
        # bin_boundaries = np.linspace(histrange[0], histrange[1], num=nbins + 1)

        # Prepare output
        weights_benchmarks_bins = np.zeros((n_bins_total, self.n_benchmarks))

        # Main loop: truth-level case
        for observations, weights in self.event_loader():

            # Cuts
            cut_filter = [self._pass_cuts(obs_event, cuts) for obs_event in observations]
            observations = observations[cut_filter]
            weights = weights[cut_filter]

            # Efficiencies
            efficiencies = np.array(
                [self._eval_efficiency(obs_event, efficiency_functions) for obs_event in observations]
            )
            weights *= efficiencies[:, np.newaxis]

            # Evaluate histogrammed observable
            histo_observables = np.asarray([self._eval_observable(obs_event, observable) for obs_event in observations])

            # Find bins
            bins = np.searchsorted(bin_boundaries, histo_observables)
            assert ((0 <= bins) & (bins < n_bins_total)).all(), "Wrong bin {}".format(bins)

            # Add up
            for i in range(n_bins_total):
                if len(weights[bins == i]) > 0:
                    weights_benchmarks_bins[i] += np.sum(weights[bins == i], axis=0)

        # Get morphing matrices
        theta_matrix = self._get_theta_benchmark_matrix(theta, zero_pad=False)  # (n_benchmarks_phys,)
        dtheta_matrix = self._get_dtheta_benchmark_matrix(theta, zero_pad=False)  # (n_parameters, n_benchmarks_phys)

        # Calculate xsecs in bins
        sigma_bins = mdot(theta_matrix, weights_benchmarks_bins)  # (n_bins,)
        dsigma_bins = mdot(dtheta_matrix, weights_benchmarks_bins)  # (n_parameters,n_bins,)

        return bin_boundaries, sigma_bins, dsigma_bins

    def nuisance_constraint_information(self):
        """Builds the Fisher information term representing the Gaussian constraints on the nuisance parameters"""

        return np.diag([0.0] * self.n_parameters + [1.0] * self.n_nuisance_parameters)

    def _check_binning_stats(
        self,
        weights_benchmarks,
        weights_benchmark_uncertainties,
        theta,
        report=5,
        n_bins_last_axis=None,
    ):
        theta_matrix = self._get_theta_benchmark_matrix(theta, zero_pad=False)  # (n_benchmarks_phys,)
        sigma = mdot(theta_matrix, weights_benchmarks)  # Shape (n_bins,)
        sigma_uncertainties = mdot(theta_matrix, weights_benchmark_uncertainties)  # Shape (n_bins,)
        rel_uncertainties = sigma_uncertainties / np.maximum(sigma, 1.0e-12)

        order = np.argsort(rel_uncertainties)[::-1]

        logger.info("Bins with largest statistical uncertainties on rates:")
        for i_bin in order[:report]:
            bin_nd = i_bin + 1
            if n_bins_last_axis is not None:
                bin_nd = (i_bin // n_bins_last_axis + 1, i_bin % n_bins_last_axis + 1)
            logger.info(
                "  Bin %s: (%.5f +/- %.5f) fb (%.0f %%)",
                bin_nd,
                1000.0 * sigma[i_bin],
                1000.0 * sigma_uncertainties[i_bin],
                100.0 * rel_uncertainties[i_bin],
            )

    def _calculate_binning(
        self,
        bins,
        cuts,
        efficiency_functions,
        histrange,
        n_events_dynamic_binning,
        observable,
        theta,
    ):
        dynamic_binning = histrange is None and isinstance(bins, int)
        if dynamic_binning:
            n_bins_total = bins
            bin_boundaries = self._calculate_dynamic_binning(
                observable, theta, bins, n_events_dynamic_binning, cuts, efficiency_functions
            )
            logger.debug("Automatic dynamic binning: bin boundaries %s", bin_boundaries)
        elif isinstance(bins, int):
            n_bins_total = bins + 2
            bin_boundaries = np.linspace(histrange[0], histrange[1], num=bins + 1)
        else:
            bin_boundaries = bins
            n_bins_total = len(bins) + 1

        return bin_boundaries, n_bins_total

    def _calculate_fisher_information(
        self,
        theta,
        weights_benchmarks,
        luminosity=300000.0,
        include_nuisance_parameters=True,
        sum_events=False,
        calculate_uncertainty=False,
        weights_benchmark_uncertainties=None,
    ):
        """
        Low-level function that calculates a list of full Fisher information matrices for a given parameter point and
        benchmark weights. Do not use this function directly, instead use the other `FisherInformation` functions.

        Parameters
        ----------
        theta : ndarray
            Parameter point.

        weights_benchmarks : ndarray
            Benchmark weights.  Shape (n_events, n_benchmark).

        luminosity : float, optional
            Luminosity in pb^-1. Default value: 300000.

        include_nuisance_parameters : bool, optional
            If True, nuisance parameters are taken into account. Default value: True.

        sum_events : bool, optional
            If True, returns the summed FIsher information. Otherwise, a list of Fisher
            information matrices for each event. Default value: False.

        calculate_uncertainty : bool, optional
            Whether an uncertainty of the result is calculated. Note that this uncertainty is currently only
            implemented for the "physical" part of the FIsher information, not for the nuisance parameters. Default
            value: False.

        weights_benchmark_uncertainties : ndarray or None, optional
            If calculate_uncertainty is True, weights_benchmark_uncertainties sets the uncertainties on each entry of
            weights_benchmarks. If None, weights_benchmark_uncertainties = weights_benchmarks is assumed.

        Returns
        -------
        fisher_information : ndarray
            If sum_events is True, the return value is an nxn matrix, the total Fisher information
            summed over all events. Otherwise, a n_events x n_parameters x n_parameters tensor is returned that
            includes the Fisher information matrices for each event separately.

        fisher_information_uncertainty : ndarray
            Only returned if calculate_uncertainty is True. Covariance matrix of the Fisher information. Note that this
            does not take into account any uncertainty on the nuisance parameter part of the Fisher information, and
            correlations between events are neglected. Note that independent of sum_events, the covariance matrix is
            always summed over the events.
        """

        # Get morphing matrices
        theta_matrix = self._get_theta_benchmark_matrix(theta, zero_pad=False)  # (n_benchmarks_phys,)
        dtheta_matrix = self._get_dtheta_benchmark_matrix(theta, zero_pad=False)  # (n_parameters, n_benchmarks_phys)

        # Get differential xsec per event, and the derivative wrt to theta
        sigma = mdot(theta_matrix, weights_benchmarks)  # Shape (n_events,)
        inv_sigma = sanitize_array(1.0 / sigma)  # Shape (n_events,)
        dsigma = mdot(dtheta_matrix, weights_benchmarks)  # Shape (n_parameters, n_events)

        # Calculate physics Fisher info for this event
        fisher_info_phys = luminosity * np.einsum("n,in,jn->nij", inv_sigma, dsigma, dsigma)

        if include_nuisance_parameters and self.nuisance_morpher is None:
            logger.warning("Cannot include nuisance parameters as none were found in the setup file")

        # Nuisance parameter Fisher info
        if include_nuisance_parameters and self.nuisance_morpher is not None:
            nuisance_a = self.nuisance_morpher.calculate_a(weights_benchmarks)  # Shape (n_nuisance_params, n_events)
            # grad_i dsigma(x), where i is a nuisance parameter, is given by
            # sigma[np.newaxis, :] * a

            fisher_info_nuisance = luminosity * np.einsum("n,in,jn->nij", sigma, nuisance_a, nuisance_a)
            fisher_info_mix = luminosity * np.einsum("in,jn->nij", dsigma, nuisance_a)
            fisher_info_mix_transposed = luminosity * np.einsum("in,jn->nji", dsigma, nuisance_a)

            n_all_parameters = self.n_parameters + self.n_nuisance_parameters
            fisher_info = np.zeros((fisher_info_phys.shape[0], n_all_parameters, n_all_parameters))
            fisher_info[:, : self.n_parameters, : self.n_parameters] = fisher_info_phys
            fisher_info[:, : self.n_parameters, self.n_parameters :] = fisher_info_mix
            fisher_info[:, self.n_parameters :, : self.n_parameters] = fisher_info_mix_transposed
            fisher_info[:, self.n_parameters :, self.n_parameters :] = fisher_info_nuisance

        else:
            n_all_parameters = self.n_parameters
            fisher_info = fisher_info_phys

        # Error propagation
        if calculate_uncertainty:
            if weights_benchmarks.shape[1] > self.n_benchmarks_phys:
                weights_benchmarks_phys = weights_benchmarks[:, np.logical_not(self.benchmark_nuisance_flags)]
            else:
                weights_benchmarks_phys = weights_benchmarks

            n_events = weights_benchmarks_phys.shape[0]

            # Input uncertainties
            if weights_benchmark_uncertainties is None:
                weights_benchmark_uncertainties = weights_benchmarks_phys  # Shape (n_events, n_benchmarks_phys)

            # Build covariance matrix of inputs
            # We assume full correlation between weights_benchmarks[i, b1] and weights_benchmarks[i, b2]
            covariance_inputs = np.zeros((n_events, self.n_benchmarks_phys, self.n_benchmarks_phys))
            for i in range(n_events):
                for b1 in range(self.n_benchmarks_phys):
                    for b2 in range(self.n_benchmarks_phys):

                        if b1 == b2:  # Diagonal
                            covariance_inputs[i, b1, b2] = weights_benchmark_uncertainties[i, b1] ** 2

                        else:  # Off-diagonal, same event
                            covariance_inputs[i, b1, b2] = (
                                weights_benchmark_uncertainties[i, b1] * weights_benchmark_uncertainties[i, b2]
                            )

            # Jacobian
            temp1 = np.einsum("ib,jn,n->ijnb", dtheta_matrix, dsigma, inv_sigma)
            temp2 = np.einsum("jb,in,n->ijnb", dtheta_matrix, dsigma, inv_sigma)
            temp3 = np.einsum("b,in,jn,n,n->ijnb", theta_matrix, dsigma, dsigma, inv_sigma, inv_sigma)

            temp1, temp2, temp3 = sanitize_array(temp1), sanitize_array(temp2), sanitize_array(temp3)

            jacobian = luminosity * (temp1 + temp2 + temp3)  # (n_parameters, n_parameters, n_events, n_benchmarks_phys)

            # Covariance of information
            covariance_information_phys = np.einsum("ijnb,nbc,klnc->ijkl", jacobian, covariance_inputs, jacobian)

            if include_nuisance_parameters:
                covariance_information = np.zeros(
                    (n_all_parameters, n_all_parameters, n_all_parameters, n_all_parameters)
                )
                covariance_information[
                    : self.n_parameters, : self.n_parameters, : self.n_parameters, : self.n_parameters
                ] = covariance_information_phys
            else:
                covariance_information = covariance_information_phys

            if sum_events:
                return np.sum(fisher_info, axis=0), covariance_information
            return fisher_info, covariance_information

        if sum_events:
            return np.sum(fisher_info, axis=0)
        return fisher_info

    def _pass_cuts(self, observations, cuts=None):
        """
        Checks if an event, specified by a list of observations, passes a set of cuts.

        Parameters
        ----------
        observations : list of float
            list of float. Values of the observables for a single event.

        cuts : list of str or None, optional
            Each entry is a parseable Python expression that returns a bool (True if the event should pass a cut,
            False otherwise). Default value: None.

        Returns
        -------
        passes : bool
            True if the event passes all cuts, False otherwise.

        """

        # Check inputs
        if cuts is None:
            cuts = []

        assert len(observations) == len(self.observables), "Mismatch between observables and observations"

        # Variables that can be used in cuts
        variables = math_commands()

        for observable_name, observable_value in zip(self.observables, observations):
            variables[observable_name] = observable_value

        # Check cuts
        for cut in cuts:
            if not bool(eval(cut, variables)):
                return False

        return True

    def _eval_efficiency(self, observations, efficiency_functions=None):
        """
        Calculates the efficiency for an event.

        Parameters
        ----------
        observations : list of float
            Values of the observables.

        efficiency_functions : list of str or None
            Each entry is a parseable Python expression that returns a float for the efficiency of one component.
            Default value: None.

        Returns
        -------
        efficiency : float
            Efficiency (0. <= efficiency <= 1.), product of the results of the calls to all entries in
            efficiency_functions.

        """

        # Check inputs
        if efficiency_functions is None:
            efficiency_functions = []

        assert len(observations) == len(self.observables), "Mismatch between observables and observations"

        # Variables that can be used in efficiency functions
        variables = math_commands()

        for observable_name, observable_value in zip(self.observables, observations):
            variables[observable_name] = observable_value

        # Check cuts
        efficiency = 1.0
        for efficency_function in efficiency_functions:
            efficiency *= float(eval(efficency_function, variables))

        return efficiency

    def _eval_observable(self, observations, observable_definition):
        """
        Calculates an observable expression for an event.

        Parameters
        ----------
        observations : ndarray
            Values of the observables for an event, should have shape `(n_observables,)`.

        observable_definition : str
            A parseable Python expression that returns the value of the observable to be calculated.

        Returns
        -------
        observable_value : float
            Value of the observable defined in observable_definition.

        """

        assert len(observations) == len(self.observables), "Mismatch between observables and observations"

        # Variables that can be used in efficiency functions
        variables = math_commands()

        for observable_name, observable_value in zip(self.observables, observations):
            variables[observable_name] = observable_value

        # Check cuts
        return float(eval(observable_definition, variables))

    def _calculate_xsec(
        self,
        theta=None,
        cuts=None,
        efficiency_functions=None,
        return_benchmark_xsecs=False,
        return_error=False,
        include_nuisance_parameters=True,
        start_event=0,
    ):
        """
        Calculates the total cross section for a parameter point.

        Parameters
        ----------
        theta : ndarray or None, optional
            The parameter point. If None, return_benchmark_xsecs should be True. Default value: None.

        cuts : list of str or None, optional
            Cuts. Each entry is a parseable Python expression that returns a bool (True if the event should pass a cut,
            False otherwise). Default value: None.

        efficiency_functions : list of str or None
            Efficiencies. Each entry is a parseable Python expression that returns a float for the efficiency of one
            component. Default value: None.

        return_benchmark_xsecs : bool, optional
            If True, this function returns the benchmark xsecs. Otherwise, it returns the xsec at theta. Default value:
            False.

        return_error : bool, optional
            If True, this function also returns the square root of the summed squared weights.

        include_nuisance_parameters : bool, optional
            If True and if return_benchmark_xsecs is True, the nuisance benchmarks are included in the output. Default
            value: True.

        start_event : int, optional
            Index of first event in MadMiner file to consider. Default value: 0.

        Returns
        -------
        xsec : ndarray or float
            If return_benchmark_xsecs is True, an ndarray of benchmark xsecs in pb is returned. Otherwise, the cross
            section at theta in pb is returned.

        xsec_uncertainty : ndarray or float
            Only returned if return_error is True. Uncertainty (square root of the summed squared weights) on xsec.

        """

        logger.debug("Calculating total cross section for theta = %s", theta)

        # Input
        if cuts is None:
            cuts = []
        if efficiency_functions is None:
            efficiency_functions = []

        assert (theta is not None) or return_benchmark_xsecs, "Please supply theta or set return_benchmark_xsecs=True"

        # Total xsecs for benchmarks
        xsecs_benchmarks = None
        xsecs_uncertainty_benchmarks = None

        for observations, weights in self.event_loader(
            start=start_event, include_nuisance_parameters=include_nuisance_parameters
        ):
            # Cuts
            cut_filter = [self._pass_cuts(obs_event, cuts) for obs_event in observations]
            observations = observations[cut_filter]
            weights = weights[cut_filter]

            # Efficiencies
            efficiencies = np.array(
                [self._eval_efficiency(obs_event, efficiency_functions) for obs_event in observations]
            )
            weights *= efficiencies[:, np.newaxis]

            # xsecs
            if xsecs_benchmarks is None:
                xsecs_benchmarks = np.sum(weights, axis=0)
                xsecs_uncertainty_benchmarks = np.sum(weights**2, axis=0)
            else:
                xsecs_benchmarks += np.sum(weights, axis=0)
                xsecs_uncertainty_benchmarks += np.sum(weights**2, axis=0)

        assert xsecs_benchmarks is not None, "No events passed cuts"

        xsecs_uncertainty_benchmarks = xsecs_uncertainty_benchmarks**0.5

        logger.debug("Benchmarks xsecs [pb]: %s", xsecs_benchmarks)

        if return_benchmark_xsecs:
            if return_error:
                return xsecs_benchmarks, xsecs_uncertainty_benchmarks
            return xsecs_benchmarks

        # Translate to xsec for theta
        theta_matrix = self._get_theta_benchmark_matrix(theta)
        xsec = mdot(theta_matrix, xsecs_benchmarks)
        xsec_error = mdot(theta_matrix, xsecs_uncertainty_benchmarks)

        logger.debug("Theta matrix: %s", theta_matrix)
        logger.debug("Cross section at theta: %s pb", xsec)

        if return_error:
            return xsec, xsec_error
        return xsec

    def _calculate_dynamic_binning(
        self,
        observable,
        theta,
        n_bins,
        n_events=None,
        cuts=None,
        efficiency_functions=None,
    ):

        if cuts is None:
            cuts = []
        if efficiency_functions is None:
            efficiency_functions = []

        # Quantile values
        quantile_values = np.linspace(0.0, 1.0, n_bins + 1)

        # Get data
        x_pilot, weights_pilot = next(self.event_loader(batch_size=n_events))

        # Cuts
        cut_filter = [self._pass_cuts(x, cuts) for x in x_pilot]
        x_pilot = x_pilot[cut_filter]
        weights_pilot = weights_pilot[cut_filter]

        # Efficiencies
        efficiencies = np.array([self._eval_efficiency(x, efficiency_functions) for x in x_pilot])
        weights_pilot *= efficiencies[:, np.newaxis]

        # Evaluate histogrammed observable
        histo_observables_pilot = np.asarray([self._eval_observable(x, observable) for x in x_pilot])

        # Weights at theta
        theta_matrix = self._get_theta_benchmark_matrix(theta)
        weight_theta_pilot = mdot(theta_matrix, weights_pilot)

        # Bin boundaries
        bin_boundaries = weighted_quantile(histo_observables_pilot, quantile_values, weight_theta_pilot)
        bin_boundaries = bin_boundaries[1:-1]

        return bin_boundaries

    # Aliases for backward compatibility
    calculate_fisher_information_full_truth = truth_information
    calculate_fisher_information_full_detector = full_information
    calculate_fisher_information_rate = rate_information
    calculate_fisher_information_hist1d = histo_information
    calculate_fisher_information_hist2d = histo_information_2d
    histogram_of_fisher_information = histogram_of_information
    calculate_fisher_information_nuisance_constraints = nuisance_constraint_information
