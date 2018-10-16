from __future__ import absolute_import, division, print_function, unicode_literals

import six
from collections import OrderedDict
import numpy as np
import logging

from madminer.utils.interfaces.hdf5 import save_events_to_madminer_file, load_benchmarks_from_madminer_file
from madminer.utils.interfaces.delphes import run_delphes
from madminer.utils.interfaces.root import extract_observables_from_delphes_file
from madminer.utils.interfaces.hepmc import extract_weight_order
from madminer.utils.various import general_init


class DelphesProcessor:
    """Detector simulation with Delphes and simple calculation of observables.
    
    After setting up the parameter space and benchmarks and running MadGraph and Pythia, all of which is organized
    in the madminer.core.MadMiner class, the next steps are the simulation of detector effects and the calculation of
    observables.  Different tools can be used for these tasks, please feel free to implement the detector simulation and
    analysis routine of your choice.
    
    This class provides an example implementation based on Delphes. Its workflow consists of four steps:
    - Initializing the class with the filename of a MadMiner HDF5 file (the output of `madminer.core.MadMiner.save()`)
    - Adding one or multiple HepMC samples produced by Pythia in `DelphesProcessor.add_hepmc_sample()`
    - Running Delphes on these samples through `DelphesProcessor.run_delphes()`
    - Defining observables through `DelphesProcessor.add_observables()`. A simple set of default observables is provided
    with `DelphesProcessor.add_default_observables()`
    - Optionally, cuts can be set with `DelphesProcessor.add_cut()`
    - Calculating the observables from the Delphes ROOT files with `DelphesProcessor.analyse_delphes_samples()`
    - Saving the results with `DelphesProcessor.save()`
    
    Please see the tutorial for a detailed walk-through.

    Parameters
    ----------

    Returns
    -------

    """

    def __init__(self, filename=None, debug=False):
        """ Constructor """

        general_init(debug=debug)

        # Initialize samples
        self.hepmc_sample_filenames = []
        self.delphes_sample_filenames = []
        self.hepmc_sample_weight_labels = []
        self.hepmc_sampled_from_benchmark = []

        # Initialize observables
        self.observables = OrderedDict()
        self.observables_required = OrderedDict()
        self.observables_defaults = OrderedDict()

        # Initialize cuts
        self.cuts = []
        self.cuts_default_pass = []

        # Initialize acceptance cuts
        self.acceptance_pt_min_e = 10.
        self.acceptance_pt_min_mu = 10.
        self.acceptance_pt_min_a = 10.
        self.acceptance_pt_min_j = 20.
        self.acceptance_eta_max_e = 2.5
        self.acceptance_eta_max_mu = 2.5
        self.acceptance_eta_max_a = 2.5
        self.acceptance_eta_max_j = 5.

        # Initialize samples
        self.observations = None
        self.weights = None

        # Information from .h5 file
        self.filename = filename
        if self.filename is None:
            self.benchmark_names = None
        else:
            self.benchmark_names = load_benchmarks_from_madminer_file(self.filename)

    def add_hepmc_sample(self, filename, sampled_from_benchmark):
        """

        Parameters
        ----------
        filename :
            
        sampled_from_benchmark :
            

        Returns
        -------

        """

        logging.info('Adding HepMC sample at %s', filename)

        self.hepmc_sample_filenames.append(filename)
        self.hepmc_sample_weight_labels.append(
            extract_weight_order(filename, sampled_from_benchmark)
        )

    def run_delphes(self, delphes_directory, delphes_card, initial_command=None, log_directory=None):
        """

        Parameters
        ----------
        delphes_directory :
            
        delphes_card :
            
        initial_command :
             (Default value = None)
        log_directory :
             (Default value = None)

        Returns
        -------

        """

        if log_directory is None:
            log_directory = './logs'
        log_file = log_directory + '/delphes.log'

        for hepmc_sample_filename in self.hepmc_sample_filenames:
            logging.info('Running Delphes (%s) on event sample at %s', delphes_directory, hepmc_sample_filename)
            delphes_sample_filename = run_delphes(
                delphes_directory,
                delphes_card,
                hepmc_sample_filename,
                initial_command=initial_command,
                log_file=log_file
            )
            self.delphes_sample_filenames.append(delphes_sample_filename)

    def add_delphes_sample(self, filename):
        """

        Parameters
        ----------
        filename :
            

        Returns
        -------

        """

        raise NotImplementedError('Direct use of Delphes samples is currently disabled since the Delphes file alone '
                                  'does not contain any information linking the weights to the benchmarks ')

        # logging.info('Adding Delphes sample at %s', filename)
        # self.delphes_sample_filenames.append(filename)

    def set_acceptance(self, pt_min_e=10., pt_min_mu=10., pt_min_a=0., pt_min_j=20.,
                       eta_max_e=2.5, eta_max_mu=2.5, eta_max_a=2.5, eta_max_j=5.):
        """

        Parameters
        ----------
        pt_min_e :
             (Default value = 10.)
        pt_min_mu :
             (Default value = 10.)
        pt_min_a :
             (Default value = 0.)
        pt_min_j :
             (Default value = 20.)
        eta_max_e :
             (Default value = 2.5)
        eta_max_mu :
             (Default value = 2.5)
        eta_max_a :
             (Default value = 2.5)
        eta_max_j :
             (Default value = 5.)

        Returns
        -------

        """

        self.acceptance_pt_min_e = pt_min_e
        self.acceptance_pt_min_mu = pt_min_mu
        self.acceptance_pt_min_a = pt_min_a
        self.acceptance_pt_min_j = pt_min_j
        self.acceptance_eta_max_e = eta_max_e
        self.acceptance_eta_max_mu = eta_max_mu
        self.acceptance_eta_max_a = eta_max_a
        self.acceptance_eta_max_j = eta_max_j

    def add_observable(self, name, definition, required=False, default=None):
        """

        Parameters
        ----------
        name :
            
        definition :
            
        required :
             (Default value = False)
        default :
             (Default value = None)

        Returns
        -------

        """

        if required:
            logging.debug('Adding required observable %s = %s', name, definition)
        else:
            logging.debug('Adding optional observable %s = %s with default %s', name, definition, default)

        self.observables[name] = definition
        self.observables_required[name] = required
        self.observables_defaults[name] = default

    def read_observables_from_file(self, filename):
        """

        Parameters
        ----------
        filename :
            

        Returns
        -------

        """
        raise NotImplementedError

    def add_default_observables(
            self,
            n_leptons_max=2,
            n_photons_max=2,
            n_jets_max=2,
            include_met=True
    ):
        """

        Parameters
        ----------
        n_leptons_max :
             (Default value = 2)
        n_photons_max :
             (Default value = 2)
        n_jets_max :
             (Default value = 2)
        include_met :
             (Default value = True)

        Returns
        -------

        """
        # ETMiss
        if include_met:
            self.add_observable(
                'et_miss',
                'met.pt',
                required=True
            )
            self.add_observable(
                'phi_miss',
                'met.phi()',
                required=True
            )

        # Observed particles
        for n, symbol in zip([n_leptons_max, n_photons_max, n_jets_max], ['l', 'a', 'j']):
            self.add_observable(
                'n_{}s'.format(symbol),
                'len({})'.format(symbol),
                required=True
            )

            for i in range(n):
                self.add_observable(
                    'e_{}{}'.format(symbol, i + 1),
                    '{}[{}].pt'.format(symbol, i),
                    required=False,
                    default=0.
                )
                self.add_observable(
                    'pt_{}{}'.format(symbol, i + 1),
                    '{}[{}].e'.format(symbol, i),
                    required=False,
                    default=0.
                )
                self.add_observable(
                    'eta_{}{}'.format(symbol, i + 1),
                    '{}[{}].eta'.format(symbol, i),
                    required=False,
                    default=0.
                )
                self.add_observable(
                    'phi_{}{}'.format(symbol, i + 1),
                    '{}[{}].phi()'.format(symbol, i),
                    required=False,
                    default=0.
                )

    def add_cut(self, definition, pass_if_not_parsed=False):
        """

        Parameters
        ----------
        definition :
            
        pass_if_not_parsed :
             (Default value = False)

        Returns
        -------

        """
        logging.info('Adding cut %s', definition)
        self.cuts.append(definition)
        self.cuts_default_pass.append(pass_if_not_parsed)

    def analyse_delphes_samples(self):
        """ """

        n_benchmarks = None if self.benchmark_names is None else len(self.benchmark_names)

        for delphes_file, weight_labels in zip(self.delphes_sample_filenames, self.hepmc_sample_weight_labels):

            logging.info('Analysing Delphes sample %s', delphes_file)

            # Calculate observables and weights in Delphes ROOT file
            this_observations, this_weights = extract_observables_from_delphes_file(
                delphes_file,
                self.observables,
                self.observables_required,
                self.observables_defaults,
                self.cuts,
                self.cuts_default_pass,
                weight_labels,

            )

            # Number of benchmarks
            if n_benchmarks is None:
                n_benchmarks = len(this_weights)

            # Background scenario: we only have one set of weights, but these should be true for all benchmarks
            if len(this_weights) == 1 and self.benchmark_names is not None:
                original_weights = list(six.itervalues(this_weights))[0]

                this_weights = OrderedDict()
                for benchmark_name in self.benchmark_names:
                    this_weights[benchmark_name] = original_weights

            # First results
            if self.observations is None and self.weights is None:
                self.observations = this_observations
                self.weights = this_weights
                continue

            # Following results: check consistency with previous results
            if len(self.weights) != len(this_weights):
                raise ValueError("Number of weights in different Delphes files incompatible: {} vs {}".format(
                    len(self.weights), len(this_weights)
                ))
            if len(self.observations) != len(this_observations):
                raise ValueError("Number of observations in different Delphes files incompatible: {} vs {}".format(
                    len(self.observations), len(this_observations)
                ))

            # Merge results with previous
            for key in self.weights:
                assert key in this_weights, "Weight label {} not found in Delphes sample!".format(
                    key
                )
                self.weights[key] = np.hstack([self.weights[key], this_weights[key]])

            for key in self.observations:
                assert key in this_observations, "Observable {} not found in Delphes sample!".format(
                    key
                )
                self.observations[key] = np.hstack([self.observations[key], this_observations[key]])

    def save(self, filename_out):
        """

        Parameters
        ----------
        filename_out :
            

        Returns
        -------

        """

        assert (self.observables is not None and self.observations is not None
                and self.weights is not None), 'Nothing to save!'

        if self.filename is None:
            logging.info('Saving HDF5 file to %s', filename_out)
        else:
            logging.info('Loading HDF5 data from %s and saving file to %s', self.filename, filename_out)

        save_events_to_madminer_file(filename_out,
                                     self.observables,
                                     self.observations,
                                     self.weights,
                                     copy_from=self.filename)
