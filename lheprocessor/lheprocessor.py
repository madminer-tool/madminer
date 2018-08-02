from __future__ import absolute_import, division, print_function, unicode_literals

from collections import OrderedDict
import numpy as np
import logging

#from lheprocessor.tools.h5_interface import save_madminer_file
#from lheprocessor.tools.delphes_interface import run_delphes
#from lheprocessor.tools.root_interface import extract_observables_from_delphes_file
from lheprocessor.tools.lhe_interface import extract_observables_from_lhe_file
from lheprocessor.tools.utils import general_init


class LHEProcessor:
    """ """

    def __init__(self, debug=False):
        """ Constructor """

        general_init(debug=debug)

        # Initialize samples
        self.lhe_sample_filenames = []
        self.sampling_benchmarks = []

        # Initialize observables
        self.observables = OrderedDict()
        self.observables_required = OrderedDict()

        # Initialize samples
        self.observations = None
        self.weights = None

    def add_lhe_sample(self, filename, sampling_benchmark):

        logging.info('Adding LHE sample at %s', filename)

        self.lhe_sample_filenames.append(filename)
        self.sampling_benchmarks.append(sampling_benchmark)

    def add_observable(self, name, definition, required=False):

        if required:
            logging.info('Adding required observable %s = %s', name, definition)
        else:
            logging.info('Adding (not required) observable %s = %s', name, definition)

        self.observables[name] = definition
        self.observables_required[name] = required

    def read_observables_from_file(self, filename):
        raise NotImplementedError

    def set_default_observables(self):
        raise NotImplementedError
    
    def analyse_lhe_samples(self):

        for lhe_file, sampling_benchmark in zip(self.lhe_sample_filenames, self.sampling_benchmarks):

            logging.info('Analysing LHE sample %s', lhe_file)

            # Calculate observables and weights
            this_observations, this_weights = extract_observables_from_lhe_file(
                lhe_file,
                sampling_benchmark,
                self.observables
            )

            # Merge
            if self.observations is None and self.weights is None:
                self.observations = this_observations
                self.weights = this_weights
                continue

            #Warning: 'OrderedDict' object has no attribute 'shape'
            if self.weights.shape[0] != this_weights.shape[0]:
                raise ValueError("Number of weights in different Delphes files incompatible: {} vs {}".format(
                    self.weights.shape[0], this_weights.shape[0]
                ))
            if len(self.observations) != len(this_observations):
                raise ValueError("Number of observations in different Delphes files incompatible: {} vs {}".format(
                    len(self.observations), len(this_observations)
                ))

            #Warning: this_weights in a OrderedDict. Does hstack work here?
            print ('FelixWarning: I am not sure if the hstack works for my weight format. Improve.')
            self.weights = np.hstack([self.weights, this_weights])

            for key in self.observations:
                assert key in this_observations, "Observable {} not found in LHE sample!".format(
                    key
                )
                self.observations[key] = np.hstack([self.observations[key], this_observations[key]])


    def save(self, filename_out, filename_in=None):

        assert (self.observables is not None and self.observations is not None
                and self.weights is not None), 'Nothing to save!'

        if filename_in is None:
            logging.info('Saving HDF5 file to %s', filename_out)
        else:
            logging.info('Loading HDF5 data from %s and saving file to %s', filename_in, filename_out)

        save_madminer_file(filename_out,
                           self.observables,
                           self.observations,
                           self.weights,
                           copy_from=filename_in)
    
