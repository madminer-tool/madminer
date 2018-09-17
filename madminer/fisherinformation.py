from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
import six
from collections import OrderedDict

from madminer.utils.interfaces.hdf5 import load_madminer_settings, madminer_event_loader
from madminer.utils.analysis import get_theta_benchmark_matrix, get_dtheta_benchmark_matrix
from madminer.morphing import SimpleMorpher as Morpher
from madminer.utils.utils import general_init, format_benchmark


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
        # This might be usefull for plotting histograms
        
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
        
        fisher_info=np.nan_to_num(fisher_info)
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

        # This Function returns the FULL Fisher Information for a given benchmark 'theta' and
        # luminosity, requiring that the events pass a set of cuts
        """
        :param theta: list (components of theta) of float
        :param luminosity: luminosity in fb^-1, float
        :param cuts: list (cuts) of definition of cuts (string)
        :return: fisher_info (nxn tensor)
        """

        # Get raw data
        x_raw, weights_benchmarks_raw = next(madminer_event_loader(self.madminer_filename, batch_size=None))

        # Select data that passes cuts
        #x = []
        weights_benchmarks = []
        for i in range(len(x_raw)):
            if self.passed_cuts(x_raw[i], cuts):
                # x.append(x_raw[i])
                weights_benchmarks.append(weights_benchmarks_raw[i])

        # Convert to Array
        #x=np.array(x)
        weights_benchmarks=np.array(weights_benchmarks)
        
        # Get Fisher Info
        fisher_info_events = self._calculate_fisher_information(theta, weights_benchmarks, luminosity)

        # Sum Fisher Infos
        fisher_info = sum(fisher_info_events)

        return fisher_info

    def calculate_truth_fisher_information_rate(self, theta, luminosity, cuts):
    
        # This Function returns the RATE ONLY Fisher Information for a given benchmark 'theta' and
        # luminosity, requiring that the events pass a set of cuts
        """
        :param theta: list (components of theta) of float
        :param luminosity: luminosity in fb^-1, float
        :param cuts: list (cuts) of definition of cuts (string)
        :return: fisher_info (nxn tensor)
        """
            
        # Get raw data
        x_raw, weights_benchmarks_raw = next(madminer_event_loader(self.madminer_filename, batch_size=None))
        
        # Select data that passes cuts
        #x = []
        weights_benchmarks = np.zeros(len(weights_benchmarks_raw[0]))
        for i in range(len(x_raw)):
            if self.passed_cuts(x_raw[i], cuts):
                # x.append(x_raw[i])
                weights_benchmarks += weights_benchmarks_raw[i]

        # Convert to Array
        weights_benchmarks=np.array([weights_benchmarks])
        
        # Get Fisher Info
        fisher_info_events = self._calculate_fisher_information(theta, weights_benchmarks, luminosity)
            
        # Sum Fisher Infos (shoiuld only contain one entry anyway)
        fisher_info = sum(fisher_info_events)
                        
        return fisher_info

    def calculate_truth_fisher_information_hist1d(self, theta, luminosity, cuts, observable, nbins, histrange):

        # This Function returns the Fisher Information in a 1D Histogram for a given benchmark 'theta' and
        # luminosity, requiring that the events pass a set of cuts
        """
        :param theta: list (components of theta) of float
        :param luminosity: luminosity in fb^-1, float
        :param cuts: list (cuts) of definition of cuts (string)
        :param xobservable: string (observable)
        :param xnbins: int (number of bins)
        :param xrange: (int,int) (range of histogram)
        :return: fisher_info (nxn tensor)
        """
        
        # Get raw data
        x_raw, weights_benchmarks_raw = next(madminer_event_loader(self.madminer_filename, batch_size=None))

        # Select data that passes cuts
        x = []
        weights_benchmarks = []
        for i in range(len(x_raw)):
            if self.passed_cuts(x_raw[i], cuts):
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
    
        # Convert to Array
        xobs=np.array(xobs)
        weights_benchmarks=np.array(weights_benchmarks)

        # Get 1D Histogram
        histos=[]
        for i in range(len(weights_benchmarks.T)):
            hist, _ =np.histogram(xobs, bins=nbins, range=histrange, weights=weights_benchmarks.T[i])
            histos.append(hist)
        histos=np.array(histos).T

        # Get Fisher Info
        fisher_info_events = self._calculate_fisher_information(theta, histos, luminosity)

        # Sum Fisher Infos
        fisher_info = sum(fisher_info_events)
        
        return fisher_info

    def calculate_histogram_fisher_information(self):
        raise NotImplementedError
