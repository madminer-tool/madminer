from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
import six
from collections import OrderedDict

from madminer.tools.h5_interface import load_madminer_settings, madminer_event_loader
from madminer.tools.analysis import get_theta_benchmark_matrix, get_dtheta_benchmark_matrix
from madminer.tools.morphing import Morpher
from madminer.tools.utils import general_init, format_benchmark


class MadFisher:

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

        # This Functio cuts returns the raw data: a list of observables x and weights for the morphing benchmarks
        # for each event
        x, weights_benchmarks = next(madminer_event_loader(self.madminer_filename, batch_size=None))
        return x, weights_benchmarks

    def extract_observables_and_weights(self, thetas=None):

        # This function returns a list of observables and weights for the benchmark 'theta'
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

    def calculate_fisher_information(self, theta, weights_benchmarks, luminosity):

        # This Function calculated a list of Fisher Info Tensors for a given benchmark 'theta' and luminosity
        """
        :param theta: list (components of theta) of float
        :param weights_benchmarks: list (events) of lists (morphing benchmarks) of floats
        :param luminosity: luminosity in fb^-1, float
        :return: list (events) of fisher_info (nxn tensor)
        """

        # get morphing matrices
        theta_matrix = get_theta_benchmark_matrix('morphing',
                                                  theta,
                                                  self.benchmarks,
                                                  self.morpher
                                                  )
        dtheta_matrix = get_dtheta_benchmark_matrix('morphing',
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
        return fisher_info

    def passed_cuts(self, observables, cuts):

        # This function checks if an events, specified by a list of observables, passes a set of cuts.
        """
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

    def calculate_truth_fisher_information_full(self, theta, luminosity, cuts):

        # This Function returns a list of observables and Fisher Info Tensors for a given benchmark 'theta' and luminosity, requiring that the events pass a set of cuts
        """
        :param theta: list (components of theta) of float
        :param luminosity: luminosity in fb^-1, float
        :param cuts: list (cuts) of definition of cuts (string)
        :return: list (events) of list(observables), fisher_info (nxn tensor)
        """

        # Get raw data
        x_raw, weights_benchmarks = next(madminer_event_loader(self.madminer_filename, batch_size=None))

        # Get Fisher Info
        fisher_info_raw = self.calculate_fisher_information(theta, weights_benchmarks, luminosity)

        # cuts
        x = []
        fisher_info = []
        for i in range(len(x_raw)):
            if self.passed_cuts(x_raw[i], cuts):
                x.append(x_raw[i])
                fisher_info.append(fisher_info_raw[i])

        return x, fisher_info

    def calculate_histogram_fisher_information(self):
        raise NotImplementedError
