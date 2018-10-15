from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
import six
from collections import OrderedDict

from madminer.utils.interfaces.hdf5 import load_madminer_settings, madminer_event_loader
from madminer.utils.analysis import get_theta_benchmark_matrix, get_dtheta_benchmark_matrix
from madminer.morphing import SimpleMorpher as Morpher
from madminer.utils.various import general_init, format_benchmark, math_commands
from madminer.ml import MLForge


class FisherInformation:

    def __init__(self, filename, debug=False):

        general_init(debug=debug)

        self.madminer_filename = filename

        logging.info('Loading data from %s', filename)

        # Load data
        (self.parameters, self.benchmarks, self.morphing_components, self.morphing_matrix,
         self.observables, self.n_samples) = load_madminer_settings(filename)
        self.n_parameters = len(self.parameters)

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

    # def extract_raw_data(self):
    #     """ Returns raw observables and benchmark weights in MadMiner file """
    #
    #     x, weights_benchmarks = next(madminer_event_loader(self.madminer_filename, batch_size=None))
    #     return x, weights_benchmarks

    def extract_observables_and_weights(self, thetas=None):
        """
        Extracts observables and weights for a list of parameter points.

        :param thetas: list (theta) of list (components of theta) of float
        :return: list (events) of list (observables) of float, list (event) of list (theta) of float
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

        return x, weights_thetas

    def _calculate_fisher_information(self, theta, weights_benchmarks, luminosity=300000., sum_events=False):
        """
        Calculates a list of Fisher info matrices for a given theta and luminosity

        :param theta: ndarray. Parameter point.
        :param weights_benchmarks: list of ndarrays. Benchmark weights for all events.
        :param luminosity: float. Luminosity in pb^-1.
        :param sum_events: bool. If True, returns the summed FIsher information. Otherwise, a list of Fisher
                           information matrices for each event.
        :return: ndarray. If sum_events is True, the return value is an nxn matrix, the total Fisher information
                          summed over all events. Otherwise, a n_events x n x n tensor is returned that includes the
                          Fisher information matrices for each event separately.
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

        # Get theta, dtheta for this events
        sigma = theta_matrix.dot(weights_benchmarks.T)
        dsigma = dtheta_matrix.dot(weights_benchmarks.T)

        # Calculate Fisher info for this event
        fisher_info = []
        for i in range(len(sigma)):
            fisher_info.append(luminosity / sigma[i] * np.tensordot(dsigma.T[i], dsigma.T[i], axes=0))
        fisher_info = np.array(fisher_info)

        fisher_info = np.nan_to_num(fisher_info)

        if sum_events:
            return np.sum(fisher_info, axis=0)
        return fisher_info

    def _pass_cuts(self, observations, cuts=None):
        """
        Checks if an event, specified by a list of observables, passes a set of cuts.

        :param observations: list of float. Values of the observables for a single event.
        :param cuts: None or list strings. Each entry is a parseable Python expression that returns a bool
                     (True if the event should pass a cut, False otherwise).
        :return: True if the event passes all cuts, False otherwise.
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
        Calculated the efficiency for an event.
        
        :param observations: list of float. Values of the observables.
        :param efficiency_functions: None or list strings. Each entry is a parseable Python expression that returns a
                                     float for the efficiency of one component.
        :return: float. Product of all efficiencies.
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

    def _calculate_xsec(self, theta=None, cuts=None, efficiency_functions=None, return_benchmark_xsecs=False):
        """
        Calculates the total cross section for a parameter point.

        :param theta: ndarray. The parameter point.
        :param cuts: None or list strings. Each entry is a parseable Python expression that returns a bool
                     (True if the event should pass a cut, False otherwise).
        :param efficiency_functions: None or list strings. Each entry is a parseable Python expression that returns a
                                     float for the efficiency of one component.
        :param return_benchmark_xsecs: bool. If True, this function returns the benchmark xsecs. Otherwise, it returns
                                       the xsec at theta.
        :return: ndarray or float. If return_benchmark_xsecs is True, an ndarray of benchmark xsecs in pb is returned.
                                   Otherwise, the cross section at theta in pb is returned.
        """

        # Input
        if cuts is None:
            cuts = []
        if efficiency_functions is None:
            efficiency_functions = []

        assert (theta is not None) or return_benchmark_xsecs, 'Please supply theta or set return_benchmark_xsecs to True'

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
        Calculates the full Fisher information at parton / truth level for a given parameter point theta and
        given luminosity.

        :param theta: ndarray. The parameter point.
        :param luminosity: float. Luminosity in pb^-1.
        :param cuts: None or list strings. Each entry is a parseable Python expression that returns a bool
                     (True if the event should pass a cut, False otherwise).
        :param efficiency_functions: None or list strings. Each entry is a parseable Python expression that returns a
                                     float for the efficiency of one component.
        :return: ndarray. Total Fisher information matrix.
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
        Calculates the estimated full Fisher information at detector level for a given parameter point theta and
        given luminosity, requiring that the events pass a set of cuts

        :param theta: ndarray. The parameter point.
        :param model_file: str, filename of a trained local score regression model that was trained on samples from
                           theta (see `madminer.ml.MLForge`)
        :param unweighted_x_sample_file: str, filename of an unweighted x sample that
                                         - is sampled according to theta
                                         - obeys the cuts
                                         (see `madminer.sampling.SampleAugmenter.extract_samples_train_local()`)
        :param luminosity: float. Luminosity in pb^-1.
        :param cuts: None or list strings. Each entry is a parseable Python expression that returns a bool
                     (True if the event should pass a cut, False otherwise).
        :return: ndarray. Total estimated Fisher information matrix.
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
        Calculates the rate-only Fisher Information for a given parameter point theta and
        luminosity.

        :param theta: ndarray. The parameter point.
        :param luminosity: float. Luminosity in pb^-1.
        :param cuts: None or list strings. Each entry is a parseable Python expression that returns a bool
                     (True if the event should pass a cut, False otherwise).
        :param efficiency_functions: None or list strings. Each entry is a parseable Python expression that returns a
                                     float for the efficiency of one component.
        :return: ndarray. Rate-only Fisher information matrix.
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
            weights_benchmarks=weights_benchmarks,
            luminosity=luminosity,
            sum_events=True
        )

        return fisher_info

    # to here

    def calculate_fisher_information_hist1d(self, theta, luminosity, observable, nbins, histrange, cuts=[],
                                            efficiencies=[]):
        """
        Calculates the Fisher information in a 1D histogram for a given benchmark theta and
        luminosity, requiring that the events pass a set of cuts

        :param theta: list (components of theta) of float
        :param luminosity: luminosity in pb^-1, float
        :param cuts: list (cuts) of definition of cuts (string)
        :param observable: string (observable)
        :param nbins: int (number of bins)
        :param histrange: (int,int) (range of histogram)
        :return: fisher_info (nxn tensor)
        """

        # Get raw data
        x_raw, weights_benchmarks_raw = next(madminer_event_loader(self.madminer_filename, batch_size=None))

        # Select data that passes cuts
        x = []
        weights_benchmarks = []
        for i in range(len(x_raw)):
            if self._pass_cuts(x_raw[i], cuts):
                x.append(x_raw[i])
                event_efficiency = self._eval_efficiency(x_raw[i], efficiencies)
                weights_benchmarks.append(weights_benchmarks_raw[i] * event_efficiency)

        # Eevaluate relevant observable
        xobs = []
        for i in range(len(x)):
            event_observables = OrderedDict()
            j = 0
            for key, _ in self.observables.items():
                event_observables[key] = x[i][j]
                j += 1
            xobs.append(eval(observable, event_observables, math_commands()))

        # Convert to array
        xobs = np.array(xobs)
        weights_benchmarks = np.array(weights_benchmarks)

        # Get 1D Histogram
        raw_xbins = np.linspace(histrange[0], histrange[1], num=nbins + 1)
        use_xbins = [np.array([-np.inf]), raw_xbins, np.array([np.inf])]
        use_xbins = np.concatenate(use_xbins)
        histos = []
        for i in range(len(weights_benchmarks.T)):
            hist, _ = np.histogram(xobs, bins=use_xbins, weights=weights_benchmarks.T[i])
            histos.append(hist)
        histos = np.array(histos).T

        # Get Fisher Info
        fisher_info_events = self._calculate_fisher_information(theta, histos, luminosity)

        # Sum Fisher Infos
        fisher_info = sum(fisher_info_events)

        return fisher_info

    def calculate_fisher_information_hist2d(self, theta, luminosity, observable1, nbins1, histrange1, observable2,
                                            nbins2, histrange2, cuts=[], efficiencies=[]):
        """
        Calculates the Fisher information in a 2D histogram for a given benchmark theta and
        luminosity, requiring that the events pass a set of cuts
        
        :param theta: list (components of theta) of float
        :param luminosity: luminosity in pb^-1, float
        :param cuts: list (cuts) of definition of cuts (string)
        :param observable1: string (observable)
        :param nbins1: int (number of bins)
        :param histrange1: (int,int) (range of histogram)
        :param observable2: string (observable)
        :param nbins2: int (number of bins)
        :param histrange2: (int,int) (range of histogram)
        :return: fisher_info (nxn tensor)
        """

        # Get raw data
        x_raw, weights_benchmarks_raw = next(madminer_event_loader(self.madminer_filename, batch_size=None))

        # Select data that passes cuts
        x = []
        weights_benchmarks = []
        for i in range(len(x_raw)):
            if self._pass_cuts(x_raw[i], cuts):
                x.append(x_raw[i])
                event_efficiency = self._eval_efficiency(x_raw[i], efficiencies)
                weights_benchmarks.append(weights_benchmarks_raw[i] * event_efficiency)

        # Evaluate relevant observable
        x1obs = []
        x2obs = []
        for i in range(len(x)):
            event_observables = OrderedDict()
            j = 0
            for key, _ in self.observables.items():
                event_observables[key] = x[i][j]
                j += 1
            x1obs.append(eval(observable1, event_observables, math_commands()))
            x2obs.append(eval(observable2, event_observables, math_commands()))

        # Convert to array
        x1obs = np.array(x1obs)
        x2obs = np.array(x2obs)
        weights_benchmarks = np.array(weights_benchmarks)

        # Get 1D Histogram
        raw_xbins = np.linspace(histrange1[0], histrange1[1], num=nbins1 + 1)
        use_xbins = [np.array([-np.inf]), raw_xbins, np.array([np.inf])]
        use_xbins = np.concatenate(use_xbins)
        raw_ybins = np.linspace(histrange2[0], histrange2[1], num=nbins2 + 1)
        use_ybins = [np.array([-np.inf]), raw_ybins, np.array([np.inf])]
        use_ybins = np.concatenate(use_ybins)
        histos = []
        for i in range(len(weights_benchmarks.T)):
            hist, _, _ = np.histogram2d(x1obs, x2obs, bins=(use_xbins, use_ybins), weights=weights_benchmarks.T[i])
            histos.append(hist.flatten())
        histos = np.array(histos).T

        # Get Fisher Info
        fisher_info_events = self._calculate_fisher_information(theta, histos, luminosity)

        # Sum Fisher Infos
        fisher_info = sum(fisher_info_events)

        return fisher_info

    def ignore_information(self,
                           fisher_information_old,
                           remaining_components):
        """
        :param fisher_information_old: fisher info (size N x N)
        :param remaining_components: is list (length M) of indices (integer) of which rows / columns to keep
        :return: fisher info (size M x M)
        """
        fisher_information_new = np.zeros([len(remaining_components), len(remaining_components)])
        for xnew, xold in enumerate(remaining_components):
            for ynew, yold in enumerate(remaining_components):
                fisher_information_new[xnew, ynew] = fisher_information_old[xold, yold]
        return fisher_information_new

    def profile_information(self,
                            fisher_information,
                            remaining_components):

        """
        Calculates the profiled Fisher information matrix as defined in Appendix A.4 of 1612.05261.

        :param fisher_information: is a (N x N) numpy array with the original Fisher information.
        :param remaining_components:is list (length M) of indices (integer) of which rows / columns to keep, others are profiled over.
        :return: the profiled Fisher information as a (M x M) numpy array.
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

        if len(remaining_components) != len(remaining_components_checked):
            print('Warning: ignoring some indices in profile_information: profiled_components =', profiled_components,
                  ', using only', profiled_components_checked)

        # Sort Fisher information such that the remaining components are
        # at the beginning and the profiled at the end
        profiled_fisher_information = np.copy(fisher_information)
        for i in range(n_components):
            for j in range(n_components):
                profiled_fisher_information[i, j] = fisher_information[new_index_order[i], new_index_order[j]]

        # Profile over one component at a time
        for c in list(reversed(range(len(remaining_components), n_components))):
            profiled_fisher_information = (profiled_fisher_information[:c, :c]
                                           - np.outer(profiled_fisher_information[c, :c],
                                                      profiled_fisher_information[c, :c])
                                           / profiled_fisher_information[c, c])

        return profiled_fisher_information

    def histogram_of_fisher_information(self, theta, luminosity, observable, nbins, histrange, cuts=[],
                                        efficiencies=[]):
        """
        Calculates the Fisher information in a 1D histogram for a given benchmark theta and
        luminosity, requiring that the events pass a set of cuts
        
        :param theta: list (components of theta) of float
        :param luminosity: luminosity in pb^-1, float
        :param cuts: list (cuts) of definition of cuts (string)
        :param observable: string (observable)
        :param nbins: int (number of bins)
        :param histrange: (int,int) (range of histogram)
        :return: fisher_info (nxn tensor)
        """

        # Get raw data
        x_raw, weights_benchmarks_raw = next(madminer_event_loader(self.madminer_filename, batch_size=None))

        # Select data that passes cuts
        x = []
        weights_benchmarks = []
        for i in range(len(x_raw)):
            if self._pass_cuts(x_raw[i], cuts):
                x.append(x_raw[i])
                event_efficiency = self._eval_efficiency(x_raw[i], efficiencies)
                weights_benchmarks.append(weights_benchmarks_raw[i] * event_efficiency)

        # Eevaluate relevant observable
        xobs = []
        for i in range(len(x)):
            event_observables = OrderedDict()
            j = 0
            for key, _ in self.observables.items():
                event_observables[key] = x[i][j]
                j += 1
            xobs.append(eval(observable, event_observables, math_commands()))

        # Convert to array
        xobs = np.array(xobs)
        weights_benchmarks = np.array(weights_benchmarks)

        # Get 1D Histogram
        raw_xbins = np.linspace(histrange[0], histrange[1], num=nbins + 1)
        use_xbins = [np.array([-np.inf]), raw_xbins, np.array([np.inf])]
        use_xbins = np.concatenate(use_xbins)
        histos = []
        for i in range(len(weights_benchmarks.T)):
            hist, _ = np.histogram(xobs, bins=use_xbins, weights=weights_benchmarks.T[i])
            histos.append(hist)
        histos = np.array(histos).T

        # Get weights at theta
        theta_matrix = get_theta_benchmark_matrix(
            'morphing',
            theta,
            self.benchmarks,
            self.morpher
        )

        weights_in_histo = theta_matrix.dot(histos.T)

        # Get Fisher Info in each bin
        fisher_info_histos_rate = self._calculate_fisher_information(theta, histos, luminosity)

        # Calculate FI for each event
        fisher_info_events = self._calculate_fisher_information(theta, weights_benchmarks, luminosity)

        # Bin those FI
        # The following part is pretty dumb, I just want an empty Fisher Info ...
        fisher_info_histos_full = [fisher_info_histos_rate[i] - fisher_info_histos_rate[i] for i in
                                   range(len(use_xbins) - 1)]
        for i in range(len(xobs)):
            for j in range(len(use_xbins) - 1):
                if (xobs[i] > use_xbins[j]) and (xobs[i] < use_xbins[j + 1]):
                    fisher_info_histos_full[j] += fisher_info_events[i]
        fisher_info_histos_full = np.array(fisher_info_histos_full)

        # Sum Fisher Infos
        fisher_info = sum(fisher_info_events)

        return use_xbins, weights_in_histo, fisher_info_histos_rate, fisher_info_histos_full
