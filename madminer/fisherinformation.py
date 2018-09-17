from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
import six
from collections import OrderedDict

from madminer.utils.interfaces.hdf5 import load_madminer_settings, madminer_event_loader
from madminer.utils.analysis import get_theta_benchmark_matrix, get_dtheta_benchmark_matrix
from madminer.morphing import SimpleMorpher as Morpher
from madminer.utils.various import general_init, format_benchmark
from madminer.ml import MLForge


class FisherInformation:

    def __init__(self, filename, debug=False):

        general_init(debug=debug)

        self.madminer_filename = filename

        logging.info('Loading data from %s', filename)

        # Load data
        (self.parameters, self.benchmarks, self.morphing_components, self.morphing_matrix,
         self.observables, self.n_samples) = load_madminer_settings(filename)

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
            raise ValueError('Did not find morphing setup.')

    def extract_raw_data(self):
        # This Function returns the raw data: a list of observables x and weights
        # for the morphing benchmarks for each event
        # This might be useful for plotting histograms

        x, weights_benchmarks = next(madminer_event_loader(self.madminer_filename, batch_size=None))
        return x, weights_benchmarks

    def extract_observables_and_weights(self, thetas=None):
        # This function returns a list of observables x and weights for the benchmark 'theta'
        # This might be usefull for plotting histograms
        """
        :param thetas: list (theta) of list (components of theta) of float
        :return: list (events) of list (observables) of float, list (event) of list (theta) of float
        """

        x, weights_benchmarks = next(madminer_event_loader(self.madminer_filename, batch_size=None))

        weights_thetas = []
        for theta in thetas:
            theta_matrix = get_theta_benchmark_matrix('morphing',
                                                      theta,
                                                      self.benchmarks,
                                                      self.morpher
                                                      )
            weights_thetas.append(theta_matrix.dot(weights_benchmarks.T))

        return x, weights_thetas

    def _calculate_fisher_information(self, theta, weights_benchmarks, luminosity):
        """
        Calculates a list of Fisher info matrices for a given theta and luminosity

        :param theta: list (components of theta) of float
        :param weights_benchmarks: list (events) of lists (morphing benchmarks) of floats
        :param luminosity: luminosity in fb^-1, float
        :return: list (events) of fisher_info (nxn tensor)
        """

        # get morphing matrices
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

        # get theta, dtheta for this events
        sigma = theta_matrix.dot(weights_benchmarks.T)
        dsigma = dtheta_matrix.dot(weights_benchmarks.T)

        # calculate fisher info for this event
        fisher_info = []
        for i in range(len(sigma)):
            fisher_info.append(luminosity / sigma[i] * np.tensordot(dsigma.T[i], dsigma.T[i], axes=0))

        fisher_info = np.nan_to_num(fisher_info)
        return fisher_info

    def _pass_cuts(self, observables, cuts):
        """
        Checks if an event, specified by a list of observables, passes a set of cuts.

        :param observables: list (observables) of float
        :param cuts: list (cuts) of definition of cuts (string)
        :return: True if the event passes all cuts, False otherwise
        """

        event_observables = OrderedDict()
        i = 0
        for key, _ in self.observables.items():
            event_observables[key] = observables[i]
            i += 1

        for cut in cuts:
            if not eval(cut, event_observables):
                return False
        return True

    def _calculate_xsec(self, theta, cuts):
        """
        Calculates the total cross section for a parameter point.

        :param theta: ndarray specifying the parameter point.
        :param theta: list (cuts) of definition of cuts (string)
        :return: float, cross section in pb
        """

        # Total xsecs for benchmarks
        xsecs_benchmarks = None

        for observations, weights in madminer_event_loader(self.madminer_filename):
            cut_filter = [self._pass_cuts(obs_event, cuts) for obs_event in observations]
            if xsecs_benchmarks is None:
                xsecs_benchmarks = np.sum(weights[cut_filter], axis=0)
            else:
                xsecs_benchmarks += np.sum(weights[cut_filter], axis=0)

        # Translate to xsec for theta
        theta_matrix = get_theta_benchmark_matrix(
            'morphing',
            theta,
            self.benchmarks,
            self.morpher
        )
        xsec = theta_matrix.dot(xsecs_benchmarks)

        return xsec

    def calculate_fisher_information_full_truth(self, theta, luminosity, cuts):
        """
        Calculates the full Fisher information at the parton level for a given parameter point theta and
        given luminosity, requiring that the events pass a set of cuts

        :param theta: list (components of theta) of float
        :param luminosity: luminosity in fb^-1, float
        :param cuts: list (cuts) of definition of cuts (string)
        :return: fisher_info (nxn tensor)
        """

        # Get raw data
        x_raw, weights_benchmarks_raw = next(madminer_event_loader(self.madminer_filename, batch_size=None))

        # Select data that passes cuts
        weights_benchmarks = []
        for i in range(len(x_raw)):
            if self._pass_cuts(x_raw[i], cuts):
                weights_benchmarks.append(weights_benchmarks_raw[i])

        # Convert to array
        weights_benchmarks = np.array(weights_benchmarks)

        # Get Fisher info
        fisher_info_events = self._calculate_fisher_information(theta, weights_benchmarks, luminosity)

        # Sum Fisher Infos
        fisher_info = sum(fisher_info_events)

        return fisher_info

    def calculate_fisher_information_full_detector(self, theta, luminosity, cuts,
                                                   model_file, unweighted_x_sample_file):
        """
        Calculates the full Fisher information at detector level for a given parameter point theta and
        given luminosity, requiring that the events pass a set of cuts

        :param theta: list (components of theta) of float
        :param luminosity: luminosity in fb^-1, float
        :param cuts: list (cuts) of definition of cuts (string)
        :param model_file: str, filename of a trained local score regression model that was trained on samples from
                           theta (see `madminer.ml.MLForge`)
        :param unweighted_x_sample_file: str, filename of an unweighted x sample that
                                         - is sampled according to theta
                                         - obeys the cuts
                                         (see `madminer.sampling.SampleAugmenter.extract_samples_train_local()`)
        :return: fisher_info (nxn tensor)
        """

        # TODO: automate extraction of unweighted x sample through
        # TODO: madminer.sampling.SampleAugmenter.extract_samples_train_local()?

        # Rate part of Fisher information
        fisher_info_rate = self.calculate_fisher_information_rate(theta, luminosity, cuts)
        total_xsec = self._calculate_xsec(theta, cuts)

        # Kinematic part of Fisher information
        model = MLForge()
        model.load(model_file)
        fisher_info_kin = model.calculate_fisher_information(unweighted_x_sample_file, n_events=luminosity*total_xsec)

        return fisher_info_rate + fisher_info_kin

    def calculate_fisher_information_rate(self, theta, luminosity, cuts):
        """
        Calculates the RATE-ONLY Fisher Information for a given parameter point theta and
        luminosity, requiring that the events pass a set of cuts

        :param theta: list (components of theta) of float
        :param luminosity: luminosity in fb^-1, float
        :param cuts: list (cuts) of definition of cuts (string)
        :return: fisher_info (nxn tensor)
        """

        # Get raw data
        x_raw, weights_benchmarks_raw = next(madminer_event_loader(self.madminer_filename, batch_size=None))

        # Select data that passes cuts
        weights_benchmarks = np.zeros(len(weights_benchmarks_raw[0]))
        for i in range(len(x_raw)):
            if self._pass_cuts(x_raw[i], cuts):
                weights_benchmarks += weights_benchmarks_raw[i]

        # Convert to array
        weights_benchmarks = np.array([weights_benchmarks])

        # Get Fisher info
        fisher_info_events = self._calculate_fisher_information(theta, weights_benchmarks, luminosity)

        # Sum Fisher infos (shoiuld only contain one entry anyway)
        fisher_info = sum(fisher_info_events)

        return fisher_info

    def calculate_fisher_information_hist1d(self, theta, luminosity, cuts, observable, nbins, histrange):
        """
        Calculates the Fisher information in a 1D histogram for a given benchmark theta and
        luminosity, requiring that the events pass a set of cuts

        :param theta: list (components of theta) of float
        :param luminosity: luminosity in fb^-1, float
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
                weights_benchmarks.append(weights_benchmarks_raw[i])

        # Eevaluate relevant observable
        xobs = []
        for i in range(len(x)):
            event_observables = OrderedDict()
            j = 0
            for key, _ in self.observables.items():
                event_observables[key] = x[i][j]
                j += 1
            xobs.append(eval(observable, event_observables))

        # Convert to array
        xobs = np.array(xobs)
        weights_benchmarks = np.array(weights_benchmarks)

        # Get 1D Histogram
        histos = []
        for i in range(len(weights_benchmarks.T)):
            hist, _ = np.histogram(xobs, bins=nbins, range=histrange, weights=weights_benchmarks.T[i])
            histos.append(hist)
        histos = np.array(histos).T

        # Get Fisher Info
        fisher_info_events = self._calculate_fisher_information(theta, histos, luminosity)

        # Sum Fisher Infos
        fisher_info = sum(fisher_info_events)

        return fisher_info
