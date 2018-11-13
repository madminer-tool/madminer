from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
import six
import os

from madminer.utils.interfaces.hdf5 import load_madminer_settings, madminer_event_loader
from madminer.utils.analysis import get_theta_benchmark_matrix, get_dtheta_benchmark_matrix
from madminer.morphing import Morpher
from madminer.utils.various import general_init, format_benchmark, math_commands, weighted_quantile, sanitize_array
from madminer.ml import MLForge, EnsembleForge


def project_information(fisher_information, remaining_components):
    """
    Calculates projections of a Fisher information matrix, that is, "deletes" the rows and columns corresponding to
    some parameters not of interest.

    Parameters
    ----------
    fisher_information : ndarray
        Original n x n Fisher information.

    remaining_components : list of int
        List with m entries, each an int with 0 <= remaining_compoinents[i] < n. Denotes which parameters are kept, and
        their new order. All other parameters or projected out.

    Returns
    -------
    projected_fisher_information : ndarray
        Projected m x m Fisher information, where the `i`-th row or column corresponds to the
        `remaining_components[i]`-th row or column of fisher_information.

    """
    n_new = len(remaining_components)
    fisher_information_new = np.zeros([n_new, n_new])

    for xnew, xold in enumerate(remaining_components):
        for ynew, yold in enumerate(remaining_components):
            fisher_information_new[xnew, ynew] = fisher_information[xold, yold]

    return fisher_information_new


def profile_information(fisher_information, remaining_components):
    """
    Calculates the profiled Fisher information matrix as defined in Appendix A.4 of arXiv:1612.05261.

    Parameters
    ----------
    fisher_information : ndarray
        Original n x n Fisher information.

    remaining_components : list of int
        List with m entries, each an int with 0 <= remaining_compoinents[i] < n. Denotes which parameters are kept, and
        their new order. All other parameters or projected out.

    Returns
    -------
    profiled_fisher_information : ndarray
        Profiled m x m Fisher information, where the `i`-th row or column corresponds to the
        `remaining_components[i]`-th row or column of fisher_information.

    """

    # Group components
    n_components = len(fisher_information)
    remaining_components_checked = []
    profiled_components = []

    for i in range(n_components):
        if i in remaining_components:
            remaining_components_checked.append(i)
        else:
            profiled_components.append(i)
    new_index_order = remaining_components + profiled_components

    assert len(remaining_components) == len(remaining_components_checked), "Inconsistent input"

    # Sort Fisher information such that the remaining components are  at the beginning and the profiled at the end
    profiled_fisher_information = np.copy(fisher_information[new_index_order, :])
    profiled_fisher_information = profiled_fisher_information[:, new_index_order]

    # Profile over one component at a time
    for c in range(n_components - 1, len(remaining_components) - 1, -1):
        profiled_fisher_information = (
            profiled_fisher_information[:c, :c]
            - np.outer(profiled_fisher_information[c, :c], profiled_fisher_information[c, :c])
            / profiled_fisher_information[c, c]
        )

    return profiled_fisher_information


class FisherInformation:
    """
    Functions to calculate expected Fisher information matrices.

    After inializing a `FisherInformation` instance with the filename of a MadMiner file, different information matrices
    can be calculated:

    * `FisherInformation.calculate_fisher_information_full_truth()` calculates the full truth-level Fisher information.
      This is the information in an idealized measurement where all parton-level particles with their charges, flavours,
      and four-momenta can be accessed with perfect accuracy.
    * `FisherInformation.calculate_fisher_information_full_detector()` calculates the full Fisher information in
      realistic detector-level observations, estimated with neural networks. In addition to the MadMiner file, this
      requires a trained SALLY or SALLINO estimator as well as an unweighted evaluation sample.
    * `FisherInformation.calculate_fisher_information_rate()` calculates the Fisher information in the total cross
      section.
    * `FisherInformation.calculate_fisher_information_hist1d()` calculates the Fisher information in the histogram of
      one (parton-level or detector-level) observable.
    * `FisherInformation.calculate_fisher_information_hist2d()` calculates the Fisher information in a two-dimensional
      histogram of two (parton-level or detector-level) observables.
    * `FisherInformation.histogram_of_fisher_information()` calculates the full truth-level Fisher information in
      different slices of one observable (the "distribution of the Fisher information").

    Parameters
    ----------
    filename : str
        Path to MadMiner file (for instance the output of `madminer.delphes.DelphesProcessor.save()`).

    debug : bool, optional
        If True, additional detailed debugging output is printed. Default value: False.

    """

    def __init__(self, filename, debug=False):

        general_init(debug=debug)
        self.debug = debug

        self.madminer_filename = filename

        logging.info("Loading data from %s", filename)

        # Load data
        (
            self.parameters,
            self.benchmarks,
            self.morphing_components,
            self.morphing_matrix,
            self.observables,
            self.n_samples,
        ) = load_madminer_settings(filename)
        self.n_parameters = len(self.parameters)
        self.n_benchmarks = len(self.benchmarks)

        logging.info("Found %s parameters:", len(self.parameters))
        for key, values in six.iteritems(self.parameters):
            logging.info(
                "   %s (LHA: %s %s, maximal power in squared ME: %s, range: %s)",
                key,
                values[0],
                values[1],
                values[2],
                values[3],
            )

        logging.info("Found %s benchmarks:", len(self.benchmarks))
        for key, values in six.iteritems(self.benchmarks):
            logging.info("   %s: %s", key, format_benchmark(values))

        logging.info("Found %s observables: %s", len(self.observables), ", ".join(self.observables))
        logging.info("Found %s events", self.n_samples)

        # Morphing
        self.morpher = None
        if self.morphing_matrix is not None and self.morphing_components is not None:
            self.morpher = Morpher(self.parameters)
            self.morpher.set_components(self.morphing_components)
            self.morpher.set_basis(self.benchmarks, morphing_matrix=self.morphing_matrix)

            logging.info("Found morphing setup with %s components", len(self.morphing_components))

        else:
            raise RuntimeError("Did not find morphing setup.")

    def calculate_fisher_information_full_truth(self, theta, luminosity=300000.0, cuts=None, efficiency_functions=None):
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

        # Loop over batches
        fisher_info = np.zeros((self.n_parameters, self.n_parameters))
        covariance = np.zeros((self.n_parameters, self.n_parameters, self.n_parameters, self.n_parameters))

        for observations, weights in madminer_event_loader(self.madminer_filename):
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
                theta, weights, luminosity, sum_events=True, calculate_uncertainty=True
            )
            fisher_info += this_fisher_info
            covariance += this_covariance

        return fisher_info, covariance

    def calculate_fisher_information_full_detector(
        self,
        theta,
        model_file,
        unweighted_x_sample_file=None,
        luminosity=300000.0,
        include_xsec_info=True,
        uncertainty="ensemble",
        ensemble_vote_expectation_weight=None,
        batch_size=100000,
        test_split=0.5,
    ):
        """
        Calculates the full Fisher information in realistic detector-level observations, estimated with neural networks.
        In addition to the MadMiner file, this requires a trained SALLY or SALLINO estimator.

        Parameters
        ----------
        theta : ndarray
            Parameter point `theta` at which the Fisher information matrix `I_ij(theta)` is evaluated.

        model_file : str
            Filename of a trained local score regression model that was trained on samples from `theta` (see
            `madminer.ml.MLForge`).

        unweighted_x_sample_file : str or None
            Filename of an unweighted x sample that is sampled according to theta and obeys the cuts
            (see `madminer.sampling.SampleAugmenter.extract_samples_train_local()`). If None, the Fisher information
            is instead calculated on the full, weighted samples (the data in the MadMiner file). Default value: None.

        luminosity : float, optional
            Luminosity in pb^-1. Default value: 300000.

        include_xsec_info : bool, optional
            Whether the rate information is included in the returned Fisher information. Default value: True.

        uncertainty : {"ensemble", "expectation", "sum"}, optional
            How the covariance matrix of the Fisher information estimate is calculate. With "ensemble", the ensemble
            covariance is used. With "expectation", the expectation of the score is used as a measure of the uncertainty
            of the score estimator, and this uncertainty is propagated through to the covariance matrix. With "sum",
            both terms are summed. Default value: "ensemble".

        ensemble_vote_expectation_weight : float or list of float or None, optional
            For ensemble models, the factor that determines how much more weight is given to those estimators with small
            expectation value. If a list is given, results are returned for each element in the list. If None, or if
            `EnsembleForge.calculate_expectation()` has not been called, all estimators are treated equal. Default
            value: None.

        batch_size : int, optional
            Batch size. Default value: 100000.

        test_split : float or None, optional
            If unweighted_x_sample_file is None, this determines the fraction of weighted events used for evaluation.
            If None, all events are used (this will probably include events used during training!). Default value: 0.5.

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

        # Total xsec
        total_xsec = self._calculate_xsec(theta=theta)

        # Rate part of Fisher information
        fisher_info_rate = 0.0
        rate_covariance = 0.0
        if include_xsec_info:
            logging.info("Evaluating rate Fisher information")
            fisher_info_rate, rate_covariance = self.calculate_fisher_information_rate(
                theta=theta, luminosity=luminosity
            )

        # Load SALLY model
        if os.path.isdir(model_file):
            model_is_ensemble = True
            model = EnsembleForge(debug=self.debug)
            model.load(model_file)
        else:
            model_is_ensemble = False
            model = MLForge(debug=self.debug)
            model.load(model_file)

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
            theta_matrix = get_theta_benchmark_matrix("morphing", theta, self.benchmarks, self.morpher)

            # Prepare output
            fisher_info_kin = None
            covariance = None

            n_batches = int(np.ceil((self.n_samples - start_event) / batch_size))

            for i_batch, (observations, weights_benchmarks) in enumerate(
                madminer_event_loader(self.madminer_filename, batch_size=batch_size, start=start_event)
            ):
                logging.info("Evaluating kinematic Fisher information on batch %s / %s", i_batch + 1, n_batches)

                weights_theta = theta_matrix.dot(weights_benchmarks.T)

                # Calculate Fisher info on this batch
                if model_is_ensemble:
                    this_fisher_info, this_covariance = model.calculate_fisher_information(
                        x=observations,
                        obs_weights=weights_theta,
                        n_events=luminosity * total_xsec * np.sum(weights_theta) / total_sum_weights_theta,
                        vote_expectation_weight=ensemble_vote_expectation_weight,
                        uncertainty=uncertainty,
                    )
                else:
                    this_fisher_info = model.calculate_fisher_information(
                        x=observations, weights=weights_theta, n_events=luminosity * np.sum(weights_theta)
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
            if model_is_ensemble:
                fisher_info_kin, covariance = model.calculate_fisher_information(
                    unweighted_x_sample_file,
                    n_events=luminosity * total_xsec,
                    vote_expectation_weight=ensemble_vote_expectation_weight,
                    uncertainty=uncertainty,
                )
            else:
                fisher_info_kin = model.calculate_fisher_information(
                    unweighted_x_sample_file, n_events=luminosity * total_xsec
                )
                covariance = None

        # Returns
        if model_is_ensemble:
            if isinstance(ensemble_vote_expectation_weight, list) and len(ensemble_vote_expectation_weight) > 1:
                fisher_info_results = [
                    fisher_info_rate + this_fisher_info_kin for this_fisher_info_kin in fisher_info_kin
                ]
                covariance_results = [rate_covariance + this_covariance for this_covariance in covariance]
                return fisher_info_results, covariance_results

            else:
                return fisher_info_rate + fisher_info_kin, rate_covariance + covariance

        return fisher_info_rate + fisher_info_kin, rate_covariance

    def calculate_fisher_information_rate(self, theta, luminosity, cuts=None, efficiency_functions=None):
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

        Returns
        -------
        fisher_information : ndarray
            Expected Fisher information in the total cross section with shape `(n_parameters, n_parameters)`.

        fisher_information_uncertainty : ndarray
            Covariance matrix of the Fisher information matrix with shape
            `(n_parameters, n_parameters, n_parameters, n_parameters)`, calculated with plain Gaussian error
            propagation.

        """

        # Get weights at benchmarks
        weights_benchmarks, weights_benchmark_uncertainties = self._calculate_xsec(
            cuts=cuts, efficiency_functions=efficiency_functions, return_benchmark_xsecs=True, return_error=True
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
        )

        return fisher_info, covariance

    def calculate_fisher_information_hist1d(
        self,
        theta,
        luminosity,
        observable,
        nbins,
        histrange=None,
        cuts=None,
        efficiency_functions=None,
        n_events_dynamic_binning=100000,
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

        nbins : int
            Number of bins in the histogram, excluding overflow bins.

        histrange : tuple of float or None
            Minimum and maximum value of the histogram in the form `(min, max)`. Overflow bins are always added. If
            None, variable-width bins with equal cross section are constructed automatically

        cuts : None or list of str, optional
            Cuts. Each entry is a parseable Python expression that returns a bool (True if the event should pass a cut,
            False otherwise). Default value: None.

        efficiency_functions : list of str or None
            Efficiencies. Each entry is a parseable Python expression that returns a float for the efficiency of one
            component. Default value: None.

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

        # Automatic dynamic binning
        dynamic_binning = histrange is None
        if dynamic_binning:
            n_bins_total = nbins

            # Quantile values
            quantile_values = np.linspace(0.0, 1.0, nbins + 1)

            # Get data
            x_pilot, weights_pilot = next(
                madminer_event_loader(self.madminer_filename, batch_size=n_events_dynamic_binning)
            )

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
            theta_matrix = get_theta_benchmark_matrix("morphing", theta, self.benchmarks, self.morpher)
            weight_theta_pilot = theta_matrix.dot(weights_pilot.T)

            # Bin boundaries
            bin_boundaries = weighted_quantile(histo_observables_pilot, quantile_values, weight_theta_pilot)
            bin_boundaries = bin_boundaries[1:-1]

            logging.debug("Automatic dynamic binning: bin boundaries %s", bin_boundaries)

        # Manual binning
        else:
            n_bins_total = nbins + 2
            bin_boundaries = np.linspace(histrange[0], histrange[1], num=nbins + 1)

        # Loop over batches
        weights_benchmarks = np.zeros((n_bins_total, self.n_benchmarks))
        weights_squared_benchmarks = np.zeros((n_bins_total, self.n_benchmarks))

        for observations, weights in madminer_event_loader(self.madminer_filename):
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
                    weights_benchmarks[i] += np.sum(weights[bins == i], axis=0)
                    weights_squared_benchmarks[i] += np.sum(weights[bins == i] ** 2, axis=0)

        weights_benchmark_uncertainties = weights_squared_benchmarks ** 0.5

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

    def calculate_fisher_information_hist2d(
        self,
        theta,
        luminosity,
        observable1,
        nbins1,
        histrange1,
        observable2,
        nbins2,
        histrange2,
        cuts=None,
        efficiency_functions=None,
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

        nbins1 : int
            Number of bins along the first axis in the histogram, excluding overflow bins.

        histrange1 : tuple of float
            Minimum and maximum value of the first axis of the histogram in the form `(min, max)`. Overflow bins are
            always added.

        observable2 : str
            Expression for the first observable to be histogrammed. The str will be parsed by Python's `eval()` function
            and can use the names of the observables in the MadMiner files.

        nbins2 : int
            Number of bins along the first axis in the histogram, excluding overflow bins.

        histrange2 : tuple of float
            Minimum and maximum value of the first axis of the histogram in the form `(min, max)`. Overflow bins are
            always added.

        cuts : None or list of str, optional
            Cuts. Each entry is a parseable Python expression that returns a bool (True if the event should pass a cut,
            False otherwise). Default value: None.

        efficiency_functions : list of str or None
            Efficiencies. Each entry is a parseable Python expression that returns a float for the efficiency of one
            component. Default value: None.

        Returns
        -------
        fisher_information : ndarray
            Expected Fisher information in the histogram with shape `(n_parameters, n_parameters)`.

        """

        # Input
        if cuts is None:
            cuts = []
        if efficiency_functions is None:
            efficiency_functions = []

        # Number of bins
        n_bins1_total = nbins1 + 2
        bin1_boundaries = np.linspace(histrange1[0], histrange1[1], num=nbins1 + 1)
        n_bins2_total = nbins1 + 2
        bin2_boundaries = np.linspace(histrange2[0], histrange2[1], num=nbins2 + 1)

        # Loop over batches
        weights_benchmarks = np.zeros((n_bins1_total, n_bins2_total, self.n_benchmarks))

        for observations, weights in madminer_event_loader(self.madminer_filename):
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
            bins1 = np.searchsorted(bin1_boundaries, histo1_observables)
            bins2 = np.searchsorted(bin2_boundaries, histo2_observables)

            assert ((0 <= bins1) & (bins1 < n_bins1_total)).all(), "Wrong bin {}".format(bins1)
            assert ((0 <= bins1) & (bins1 < n_bins1_total)).all(), "Wrong bin {}".format(bins1)

            # Add up
            for i in range(n_bins1_total):
                for j in range(n_bins2_total):
                    if len(weights[(bins1 == i) & (bins2 == j)]) > 0:
                        weights_benchmarks[i, j] += np.sum(weights[(bins1 == i) & (bins2 == j)], axis=0)

        # Calculate Fisher information in histogram
        weights_benchmarks = weights_benchmarks.reshape(-1, self.n_benchmarks)
        fisher_info = self._calculate_fisher_information(theta, weights_benchmarks, luminosity, sum_events=True)

        return fisher_info

    def histogram_of_fisher_information(
        self, theta, luminosity, observable, nbins, histrange, cuts=None, efficiency_functions=None
    ):
        """
        Calculates the full and rate-only Fisher information in slices of one observable.

        Parameters
        ----------
        theta : ndarray
            Parameter point `theta` at which the Fisher information matrix `I_ij(theta)` is evaluated.

        luminosity : float
            Luminosity in pb^-1.

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

        rate_fisher_infos : ndarray
            Expected rate-only Fisher information for each slice. Has shape `(n_slices, n_parameters, n_parameters)`.

        full_fisher_infos_truth : ndarray
            Expected full truth-level Fisher information for each slice. Has shape
            `(n_slices, n_parameters, n_parameters)`.

        """

        # Input
        if cuts is None:
            cuts = []
        if efficiency_functions is None:
            efficiency_functions = []

        # Number of bins
        n_bins_total = nbins + 2
        bin_boundaries = np.linspace(histrange[0], histrange[1], num=nbins + 1)

        # Loop over batches
        weights_benchmarks_bins = np.zeros((n_bins_total, self.n_benchmarks))
        fisher_info_full_bins = np.zeros((n_bins_total, self.n_parameters, self.n_parameters))

        for observations, weights in madminer_event_loader(self.madminer_filename):
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
            histo_observables = np.asarray([self._eval_observable(obs_event, observable) for obs_event in observations])

            # Find bins
            bins = np.searchsorted(bin_boundaries, histo_observables)
            assert ((0 <= bins) & (bins < n_bins_total)).all(), "Wrong bin {}".format(bins)

            # Add up
            for i in range(n_bins_total):
                if len(weights[bins == i]) > 0:
                    weights_benchmarks_bins[i] += np.sum(weights[bins == i], axis=0)
                    fisher_info_full_bins[i] += np.sum(fisher_info_events[bins == i], axis=0)

        # Calculate xsecs in bins
        theta_matrix = get_theta_benchmark_matrix("morphing", theta, self.benchmarks, self.morpher)
        sigma_bins = theta_matrix.dot(weights_benchmarks_bins.T)  # (n_bins,)

        # Calculate rate-only Fisher informations in bins
        fisher_info_rate_bins = self._calculate_fisher_information(
            theta, weights_benchmarks_bins, luminosity, sum_events=False
        )

        return bin_boundaries, sigma_bins, fisher_info_rate_bins, fisher_info_full_bins

    def extract_raw_data(self, theta=None):

        """
        Returns all events together with the benchmark weights (if theta is None) or weights for a given theta.

        Parameters
        ----------
        theta : None or ndarray, optional
            If None, the function returns the benchmark weights. Otherwise it uses morphing to calculate the weights for
            this value of theta. Default value: None.

        Returns
        -------
        x : ndarray
            Observables with shape `(n_unweighted_samples, n_observables)`.

        weights : ndarray
            If theta is None, benchmark weights with shape  `(n_unweighted_samples, n_benchmarks)` in pb. Otherwise,
            weights for the given parameter theta with shape `(n_unweighted_samples,)` in pb.

        """

        x, weights_benchmarks = next(madminer_event_loader(self.madminer_filename, batch_size=None))

        if theta is not None:
            theta_matrix = get_theta_benchmark_matrix("morphing", theta, self.benchmarks, self.morpher)

            weights_theta = theta_matrix.dot(weights_benchmarks.T)

            return x, weights_theta

        return x, weights_benchmarks

    def extract_observables_and_weights(self, thetas):
        """
        Extracts observables and weights for given parameter points.

        Parameters
        ----------
        thetas : ndarray
            Parameter points, with shape `(n_thetas, n_parameters)`.

        Returns
        -------
        x : ndarray
            Observations `x` with shape `(n_events, n_observables)`.

        weights : ndarray
            Weights `dsigma(x|theta)` in pb with shape `(n_thetas, n_events)`.

        """

        x, weights_benchmarks = next(madminer_event_loader(self.madminer_filename, batch_size=None))

        weights_thetas = []
        for theta in thetas:
            theta_matrix = get_theta_benchmark_matrix("morphing", theta, self.benchmarks, self.morpher)
            weights_thetas.append(theta_matrix.dot(weights_benchmarks.T))

        weights_thetas = np.array(weights_thetas)

        return x, weights_thetas

    def _calculate_fisher_information(
        self,
        theta,
        weights_benchmarks,
        luminosity=300000.0,
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

        luminosity : float
            Luminosity in pb^-1.

        sum_events : bool, optional
            If True, returns the summed FIsher information. Otherwise, a list of Fisher
            information matrices for each event. Default value: False.

        Returns
        -------
        fisher_information : ndarray
            If sum_events is True, the return value is an nxn matrix, the total Fisher information
            summed over all events. Otherwise, a n_events x n_parameters x n_parameters tensor is returned that
            includes the Fisher information matrices for each event separately.

        """

        # Get morphing matrices
        theta_matrix = get_theta_benchmark_matrix("morphing", theta, self.benchmarks, self.morpher)  # (n_benchmarks,)
        dtheta_matrix = get_dtheta_benchmark_matrix(
            "morphing", theta, self.benchmarks, self.morpher
        )  # (n_parameters, n_benchmarks)

        # Get differential xsec per event, and the derivative wrt to theta
        sigma = theta_matrix.dot(weights_benchmarks.T)  # Shape (n_events,)
        inv_sigma = sanitize_array(1.0 / sigma)  # Shape (n_events,)
        dsigma = dtheta_matrix.dot(weights_benchmarks.T)  # Shape (n_parameters, n_events)

        # Calculate Fisher info for this event
        fisher_info = luminosity * np.einsum("n,in,jn->nij", inv_sigma, dsigma, dsigma)

        # # Old code:
        # fisher_info_debug = []
        # for i_event in range(len(sigma)):
        #     fisher_info_debug.append(luminosity / sigma[i_event] * np.tensordot(dsigma.T[i_event], dsigma.T[i_event],
        #                                                                         axes=0))
        # fisher_info_debug = np.array(fisher_info_debug)
        # fisher_info_debug = sanitize_array(fisher_info_debug)
        # logging.debug('Old vs new Fisher info calculation:\n%s\n%s\n%s', fisher_info - fisher_info_debug, fisher_info,
        #               fisher_info_debug)

        if calculate_uncertainty:
            n_events = weights_benchmarks.shape[0]
            n_benchmarks = weights_benchmarks.shape[1]

            # Input uncertainties
            if weights_benchmark_uncertainties is None:
                weights_benchmark_uncertainties = weights_benchmarks  # Shape (n_events, n_benchmarks)

            # Build covariance matrix of inputs
            # We assume full correlation between weights_benchmarks[i, b1] and weights_benchmarks[i, b2]
            covariance_inputs = np.zeros((n_events, n_benchmarks, n_benchmarks))
            for i, b1, b2 in zip(range(n_events), range(n_benchmarks), range(n_benchmarks)):

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

            jacobian = luminosity * (temp1 + temp2 + temp3)  # (n_parameters, n_parameters, n_events, n_benchmarks)

            # Covariance of information
            covariance_information = np.einsum("ijnb,nbc,klnc->ijkl", jacobian, covariance_inputs, jacobian)

            # Old code:
            # covariance_inputs = covariance_inputs.reshape((-1))
            # jacobian = jacobian.reshape((jacobian.shape[0] * jacobian.shape[1], -1))
            # covariance_information = np.einsum("IK,K,JK->IJ", jacobian, covariance_inputs, jacobian)
            # covariance_information = covariance_information.reshape(
            #     (self.n_parameters, self.n_parameters, self.n_parameters, self.n_parameters)
            # )

            # logging.debug('Terms in uncertainty calculation:')
            # logging.debug('  theta_matrix = %s', theta_matrix)
            # logging.debug('  dtheta_matrix = %s', dtheta_matrix)
            # logging.debug('  sigma = %s', sigma)
            # logging.debug('  inv_sigma = %s', inv_sigma)
            # logging.debug('  covariance_inputs = %s', covariance_inputs)
            # logging.debug('  temp1 = %s', temp1)
            # logging.debug('  temp2 = %s', temp2)
            # logging.debug('  temp3 = %s', temp3)
            # logging.debug('  jacobian = %s', jacobian)
            # logging.debug('  covariance_information = %s', covariance_information)

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

        # Input
        if cuts is None:
            cuts = []
        if efficiency_functions is None:
            efficiency_functions = []

        assert (theta is not None) or return_benchmark_xsecs, "Please supply theta or set return_benchmark_xsecs=True"

        # Total xsecs for benchmarks
        xsecs_benchmarks = None
        xsecs_uncertainty_benchmarks = None

        for observations, weights in madminer_event_loader(self.madminer_filename, start=start_event):
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
                xsecs_uncertainty_benchmarks = np.sum(weights ** 2, axis=0)
            else:
                xsecs_benchmarks += np.sum(weights, axis=0)
                xsecs_uncertainty_benchmarks += np.sum(weights ** 2, axis=0)

        assert xsecs_benchmarks is not None, "No events passed cuts"

        xsecs_uncertainty_benchmarks = xsecs_uncertainty_benchmarks ** 0.5

        if return_benchmark_xsecs:
            if return_error:
                return xsecs_benchmarks, xsecs_uncertainty_benchmarks
            return xsecs_benchmarks

        # Translate to xsec for theta
        theta_matrix = get_theta_benchmark_matrix("morphing", theta, self.benchmarks, self.morpher)
        xsec = theta_matrix.dot(xsecs_benchmarks)
        xsec_error = theta_matrix.dot(xsecs_uncertainty_benchmarks)

        if return_error:
            return xsec, xsec_error
        return xsec
