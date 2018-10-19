from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
import six

from madminer.utils.interfaces.hdf5 import load_madminer_settings, madminer_event_loader
from madminer.utils.analysis import get_theta_benchmark_matrix, get_dtheta_benchmark_matrix
from madminer.morphing import Morpher
from madminer.utils.various import general_init, format_benchmark, math_commands
from madminer.ml import MLForge


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
    Calculates the profiled Fisher information matrix as defined in Appendix A.4 of 1612.05261.

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
    profiled_fisher_information = np.copy(fisher_information[new_index_order, new_index_order])

    # Profile over one component at a time
    for c in reversed(range(len(remaining_components), n_components)):
        profiled_fisher_information = (profiled_fisher_information[:c, :c]
                                       - np.outer(profiled_fisher_information[c, :c],
                                                  profiled_fisher_information[c, :c])
                                       / profiled_fisher_information[c, c])

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

        self.madminer_filename = filename

        logging.info('Loading data from %s', filename)

        # Load data
        (self.parameters, self.benchmarks, self.morphing_components, self.morphing_matrix,
         self.observables, self.n_samples) = load_madminer_settings(filename)
        self.n_parameters = len(self.parameters)
        self.n_benchmarks = len(self.benchmarks)

        logging.info('Found %s parameters:', len(self.parameters))
        for key, values in six.iteritems(self.parameters):
            logging.info('   %s (LHA: %s %s, maximal power in squared ME: %s, range: %s)',
                         key, values[0], values[1], values[2], values[3])

        logging.info('Found %s benchmarks:', len(self.benchmarks))
        for key, values in six.iteritems(self.benchmarks):
            logging.info('   %s: %s',
                         key, format_benchmark(values))

        logging.info('Found %s observables: %s', len(self.observables), ', '.join(self.observables))
        logging.info('Found %s events', self.n_samples)

        # Morphing
        self.morpher = None
        if self.morphing_matrix is not None and self.morphing_components is not None:
            self.morpher = Morpher(self.parameters)
            self.morpher.set_components(self.morphing_components)
            self.morpher.set_basis(self.benchmarks, morphing_matrix=self.morphing_matrix)

            logging.info('Found morphing setup with %s components', len(self.morphing_components))

        else:
            raise RuntimeError('Did not find morphing setup.')

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
            theta_matrix = get_theta_benchmark_matrix(
                'morphing',
                theta,
                self.benchmarks,
                self.morpher
            )

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
            theta_matrix = get_theta_benchmark_matrix(
                'morphing',
                theta,
                self.benchmarks,
                self.morpher
            )
            weights_thetas.append(theta_matrix.dot(weights_benchmarks.T))

        weights_thetas = np.array(weights_thetas)

        return x, weights_thetas

    def _calculate_fisher_information(self, theta, weights_benchmarks, luminosity=300000., sum_events=False):
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
        theta_matrix = get_theta_benchmark_matrix(
            'morphing',
            theta,
            self.benchmarks,
            self.morpher
        )
        dtheta_matrix = get_dtheta_benchmark_matrix(
            'morphing',
            theta,
            self.benchmarks,
            self.morpher
        )

        # Get differential xsec per event, and the derivative wrt to theta
        sigma = theta_matrix.dot(weights_benchmarks.T)  # Shape (n_events,)
        dsigma = dtheta_matrix.dot(weights_benchmarks.T)  # Shape (n_parameters, n_events)

        # Calculate Fisher info for this event
        fisher_info = []
        for i_event in range(len(sigma)):
            fisher_info.append(luminosity / sigma[i_event] * np.tensordot(dsigma.T[i_event], dsigma.T[i_event], axes=0))
        fisher_info = np.array(fisher_info)

        fisher_info = np.nan_to_num(fisher_info)

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

        assert len(observations) == len(self.observables), 'Mismatch between observables and observations'

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

        assert len(observations) == len(self.observables), 'Mismatch between observables and observations'

        # Variables that can be used in efficiency functions
        variables = math_commands()

        for observable_name, observable_value in zip(self.observables, observations):
            variables[observable_name] = observable_value

        # Check cuts
        efficiency = 1.
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

        assert len(observations) == len(self.observables), 'Mismatch between observables and observations'

        # Variables that can be used in efficiency functions
        variables = math_commands()

        for observable_name, observable_value in zip(self.observables, observations):
            variables[observable_name] = observable_value

        # Check cuts
        return float(eval(observable_definition, variables))

    def _calculate_xsec(self, theta=None, cuts=None, efficiency_functions=None, return_benchmark_xsecs=False):
        """
        Calculates the total cross section for a parameter point.

        Parameters
        ----------
        theta : ndarray or None
            The parameter point. If None, return_benchmark_xsecs should be True. Default value: None.

        cuts : list of str or None
            Cuts. Each entry is a parseable Python expression that returns a bool (True if the event should pass a cut,
            False otherwise). Default value: None.

        efficiency_functions : list of str or None
            Efficiencies. Each entry is a parseable Python expression that returns a float for the efficiency of one
            component. Default value: None.

        return_benchmark_xsecs : bool
            If True, this function returns the benchmark xsecs. Otherwise, it returns the xsec at theta. Default value:
            False.

        Returns
        -------
        xsec : ndarray or float
            If return_benchmark_xsecs is True, an ndarray of benchmark xsecs in pb is returned. Otherwise, the cross
            section at theta in pb is returned.

        """

        # Input
        if cuts is None:
            cuts = []
        if efficiency_functions is None:
            efficiency_functions = []

        assert (theta is not None) or return_benchmark_xsecs, 'Please supply theta or set return_benchmark_xsecs=True'

        # Total xsecs for benchmarks
        xsecs_benchmarks = None

        for observations, weights in madminer_event_loader(self.madminer_filename):
            # Cuts
            cut_filter = [self._pass_cuts(obs_event, cuts) for obs_event in observations]
            observations = observations[cut_filter]
            weights = weights[cut_filter]

            # Efficiencies
            efficiencies = np.array(
                [self._eval_efficiency(obs_event, efficiency_functions) for obs_event in observations])
            weights *= efficiencies[:, np.newaxis]

            # xsecs
            if xsecs_benchmarks is None:
                xsecs_benchmarks = np.sum(weights, axis=0)
            else:
                xsecs_benchmarks += np.sum(weights, axis=0)

        assert xsecs_benchmarks is not None, "No events passed cuts"

        if return_benchmark_xsecs:
            return xsecs_benchmarks

        # Translate to xsec for theta
        theta_matrix = get_theta_benchmark_matrix(
            'morphing',
            theta,
            self.benchmarks,
            self.morpher
        )
        xsec = theta_matrix.dot(xsecs_benchmarks)

        return xsec

    def calculate_fisher_information_full_truth(self, theta, luminosity=300000., cuts=None, efficiency_functions=None):
        """
        Calculates the full Fisher information at parton / truth level. This is the information in an idealized
        measurement where all parton-level particles with their charges, flavours, and four-momenta can be accessed with
        perfect accuracy, i.e. the latent variables `z_parton` can be measured directly.

        Parameters
        ----------
        theta : ndarray
            Parameter point `theta` at which the Fisher information matrix `I_ij(theta)` is evaluated.

        luminosity : float, optional
            Luminosity in pb^-1. Default value: 300000.

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

        """

        # Input
        if cuts is None:
            cuts = []
        if efficiency_functions is None:
            efficiency_functions = []

        # Loop over batches
        fisher_info = np.zeros((self.n_parameters, self.n_parameters))

        for observations, weights in madminer_event_loader(self.madminer_filename):
            # Cuts
            cut_filter = [self._pass_cuts(obs_event, cuts) for obs_event in observations]
            observations = observations[cut_filter]
            weights = weights[cut_filter]

            # Efficiencies
            efficiencies = np.array(
                [self._eval_efficiency(obs_event, efficiency_functions) for obs_event in observations])
            weights *= efficiencies[:, np.newaxis]

            # Fisher information
            fisher_info += self._calculate_fisher_information(theta, weights, luminosity, sum_events=True)

        return fisher_info

    def calculate_fisher_information_full_detector(self, theta, model_file, unweighted_x_sample_file,
                                                   luminosity=300000., cuts=None):
        """
        Calculates the full Fisher information in realistic detector-level observations, estimated with neural networks.
        In addition to the MadMiner file, this requires a trained SALLY or SALLINO estimator as well as an unweighted
        evaluation sample.

        Parameters
        ----------
        theta : ndarray
            Parameter point `theta` at which the Fisher information matrix `I_ij(theta)` is evaluated.

        model_file : str
            str, filename of a trained local score regression model that was trained on samples from `theta` (see
            `madminer.ml.MLForge`).

        unweighted_x_sample_file : str
            str, filename of an unweighted x sample that is sampled according to theta and obeys the cuts
            (see `madminer.sampling.SampleAugmenter.extract_samples_train_local()`)

        luminosity : float, optional
            Luminosity in pb^-1. Default value: 300000.

        cuts : None or list of str, optional
            Cuts. Each entry is a parseable Python expression that returns a bool (True if the event should pass a cut,
            False otherwise). Default value: None.

        Returns
        -------
        fisher_information : ndarray
            Estimated expected full detector-level Fisher information matrix with shape `(n_parameters, n_parameters)`.

        """

        # Input
        if cuts is None:
            cuts = []

        # Rate part of Fisher information
        fisher_info_rate = self.calculate_fisher_information_rate(
            theta=theta,
            luminosity=luminosity,
            cuts=cuts
        )
        total_xsec = self._calculate_xsec(
            theta=theta,
            cuts=cuts
        )

        # Kinematic part of Fisher information
        model = MLForge()
        model.load(model_file)
        fisher_info_kin = model.calculate_fisher_information(
            unweighted_x_sample_file,
            n_events=luminosity * total_xsec
        )

        return fisher_info_rate + fisher_info_kin

    def calculate_fisher_information_rate(self, theta, luminosity, cuts=None, efficiency_functions=None):
        """
        Calculates the Fisher information in a measurement of the total cross section (without any kinematic
        information).

        Parameters
        ----------
        theta : ndarray
            Parameter point `theta` at which the Fisher information matrix `I_ij(theta)` is evaluated.

        luminosity : float, optional
            Luminosity in pb^-1. Default value: 300000.

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

        """

        # Get weights at benchmarks
        weights_benchmarks = self._calculate_xsec(
            cuts=cuts,
            efficiency_functions=efficiency_functions,
            return_benchmark_xsecs=True
        )

        # Get Fisher information
        fisher_info = self._calculate_fisher_information(
            theta=theta,
            weights_benchmarks=weights_benchmarks[np.newaxis, :],
            luminosity=luminosity,
            sum_events=True
        )

        return fisher_info

    def calculate_fisher_information_hist1d(self, theta, luminosity, observable, nbins, histrange, cuts=None,
                                            efficiency_functions=None):
        """
        Calculates the Fisher information in the one-dimensional histogram of an (parton-level or detector-level,
        depending on how the observations in the MadMiner file were calculated) observable.

        Parameters
        ----------
        theta : ndarray
            Parameter point `theta` at which the Fisher information matrix `I_ij(theta)` is evaluated.

        luminosity : float, optional
            Luminosity in pb^-1. Default value: 300000.

        observable : str
            Expression for the observable to be histogrammed. The str will be parsed by Python's `eval()` function
            and can use the names of the observables in the MadMiner files.

        nbins : int
            Number of bins in the histogram, excluding overflow bins.

        histrange : tuple of float
            Minimum and maximum value of the histogram in the form `(min, max)`. Overflow bins are always added.

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
        n_bins_total = nbins + 2
        bin_boundaries = np.linspace(histrange[0], histrange[1], num=nbins + 1)

        # Loop over batches
        weights_benchmarks = np.zeros((n_bins_total, self.n_benchmarks))

        for observations, weights in madminer_event_loader(self.madminer_filename):
            # Cuts
            cut_filter = [self._pass_cuts(obs_event, cuts) for obs_event in observations]
            observations = observations[cut_filter]
            weights = weights[cut_filter]

            # Efficiencies
            efficiencies = np.array(
                [self._eval_efficiency(obs_event, efficiency_functions) for obs_event in observations])
            weights *= efficiencies[:, np.newaxis]

            # Evaluate histogrammed observable
            histo_observables = np.asarray([self._eval_observable(obs_event, observable) for obs_event in observations])

            # Find bins
            bins = np.searchsorted(bin_boundaries, histo_observables)
            assert ((0 <= bins) & (bins < n_bins_total)).all(), 'Wrong bin {}'.format(bins)

            # Add up
            for i in range(n_bins_total):
                if len(weights[bins == i]) > 0:
                    weights_benchmarks[i] += np.sum(weights[bins == i], axis=0)

        # Calculate Fisher information in histogram
        fisher_info = self._calculate_fisher_information(theta, weights_benchmarks, luminosity, sum_events=True)

        return fisher_info

    def calculate_fisher_information_hist2d(self, theta, luminosity, observable1, nbins1, histrange1, observable2,
                                            nbins2, histrange2, cuts=None, efficiency_functions=None):

        """
        Calculates the Fisher information in a two-dimensional histogram of two (parton-level or detector-level,
        depending on how the observations in the MadMiner file were calculated) observables.

        Parameters
        ----------
        theta : ndarray
            Parameter point `theta` at which the Fisher information matrix `I_ij(theta)` is evaluated.

        luminosity : float, optional
            Luminosity in pb^-1. Default value: 300000.

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
                [self._eval_efficiency(obs_event, efficiency_functions) for obs_event in observations])
            weights *= efficiencies[:, np.newaxis]

            # Evaluate histogrammed observable
            histo1_observables = np.asarray(
                [self._eval_observable(obs_event, observable1) for obs_event in observations])
            histo2_observables = np.asarray(
                [self._eval_observable(obs_event, observable2) for obs_event in observations])

            # Find bins
            bins1 = np.searchsorted(bin1_boundaries, histo1_observables)
            bins2 = np.searchsorted(bin2_boundaries, histo2_observables)

            assert np.all(0 <= bins1 < n_bins1_total), 'Wrong bin {}'.format(bins1)
            assert np.all(0 <= bins2 < n_bins2_total), 'Wrong bin {}'.format(bins2)

            # Add up
            for i in range(n_bins1_total):
                for j in range(n_bins2_total):
                    if len(weights[(bins1 == i) & (bins2 == j)]) > 0:
                        weights_benchmarks[i, j] += np.sum(weights[(bins1 == i) & (bins2 == j)], axis=0)

        # Calculate Fisher information in histogram
        weights_benchmarks = weights_benchmarks.reshape(-1, self.n_benchmarks)
        fisher_info = self._calculate_fisher_information(theta, weights_benchmarks, luminosity, sum_events=True)

        return fisher_info

    def histogram_of_fisher_information(self, theta, luminosity, observable, nbins, histrange, cuts=None,
                                        efficiency_functions=None):
        """
        Calculates the full and rate-only Fisher information in slices of one observable.

        Parameters
        ----------
        theta : ndarray
            Parameter point `theta` at which the Fisher information matrix `I_ij(theta)` is evaluated.

        luminosity : float, optional
            Luminosity in pb^-1. Default value: 300000.

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
                [self._eval_efficiency(obs_event, efficiency_functions) for obs_event in observations])
            weights *= efficiencies[:, np.newaxis]

            # Fisher info per event
            fisher_info_events = self._calculate_fisher_information(theta, weights_benchmarks_bins, luminosity,
                                                                    sum_events=False)

            # Evaluate histogrammed observable
            histo_observables = np.asarray([self._eval_observable(obs_event, observable) for obs_event in observations])

            # Find bins
            bins = np.searchsorted(bin_boundaries, histo_observables)
            assert np.all(0 <= bins < n_bins_total), 'Wrong bin {}'.format(bins)

            # Add up
            for i in range(n_bins_total):
                if len(weights[bins == i]) > 0:
                    weights_benchmarks_bins[i] += np.sum(weights[bins == i], axis=0)
                    fisher_info_full_bins[i] += np.sum(fisher_info_events, axis=0)

        # Calculate xsecs in bins
        theta_matrix = get_theta_benchmark_matrix(
            'morphing',
            theta,
            self.benchmarks,
            self.morpher
        )
        sigma_bins = theta_matrix.dot(weights_benchmarks_bins.T)  # (n_bins,)

        # Calculate rate-only Fisher informations in bins
        fisher_info_rate_bins = self._calculate_fisher_information(theta, weights_benchmarks_bins, luminosity,
                                                                   sum_events=False)

        return bin_boundaries, sigma_bins, fisher_info_rate_bins, fisher_info_full_bins
