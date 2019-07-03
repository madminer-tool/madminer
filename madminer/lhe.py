from __future__ import absolute_import, division, print_function, unicode_literals

import six
from collections import OrderedDict
import numpy as np
import logging

from madminer.utils.interfaces.madminer_hdf5 import (
    save_events_to_madminer_file,
    load_madminer_settings,
    save_nuisance_setup_to_madminer_file,
)
from madminer.utils.interfaces.lhe import (
    parse_lhe_file,
    extract_nuisance_parameters_from_lhe_file,
    get_elementary_pdg_ids,
)

logger = logging.getLogger(__name__)


class LHEReader:
    """
    Detector simulation with smearing functions and simple calculation of observables.

    After setting up the parameter space and benchmarks and running MadGraph and Pythia, all of which is organized
    in the madminer.core.MadMiner class, the next steps are the simulation of detector effects and the calculation of
    observables. Different tools can be used for these tasks, please feel free to implement the detector simulation and
    analysis routine of your choice.

    This class provides a simple implementation in which detector effects are modeled with smearing functions. Its
    workflow consists of the following steps:

    * Initializing the class with the filename of a MadMiner HDF5 file (the output of `madminer.core.MadMiner.save()`)
    * Adding one or multiple event samples produced by MadGraph and Pythia in `LHEProcessor.add_sample()`.
    * Running Delphes on the samples that require it through `LHEProcessor.run_delphes()`.
    * Optionally, smearing functions for all visible particles can be defined with
      `LHEProcessor.set_smearing()`.
    * Defining observables through `LHEProcessor.add_observable()` or
      `LHEProcessor.add_observable_from_function()`. A simple set of default observables is provided in
      `LHEProcessor.add_default_observables()`
    * Optionally, cuts can be set with `LHEProcessor.add_cut()`
    * Optionally, efficiencies can be set with `LHEProcessor.add_efficiency()`
    * Calculating the observables from the Delphes ROOT files with `LHEProcessor.analyse_delphes_samples()`
    * Saving the results with `LHEProcessor.save()`

    Please see the tutorial for a detailed walk-through.

    Parameters
    ----------
    filename : str or None, optional
        Path to MadMiner file (the output of `madminer.core.MadMiner.save()`). Default value: None.

    """

    def __init__(self, filename):
        # Initialize samples
        self.lhe_sample_filenames = []
        self.sample_k_factors = []
        self.sample_is_backgrounds = []
        self.sampling_benchmarks = []

        # Initialize observables
        self.observables = OrderedDict()
        self.observables_required = OrderedDict()
        self.observables_defaults = OrderedDict()

        # Initialize cuts
        self.cuts = []
        self.cuts_default_pass = []

        # Initialize efficiencies
        self.efficiencies = []
        self.efficiencies_default_pass = []

        # Smearing function parameters
        self.energy_resolution = {}
        self.pt_resolution = {}
        self.eta_resolution = {}
        self.phi_resolution = {}

        pdgids = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 9]
        pdgids += [11, -11, 12, -12, 13, -13, 14, -14, 15, -15, 16, -16, 21, 22, 23, 24, -24, 25]
        for pdgid in pdgids:
            self.energy_resolution[pdgid] = (0.0, 0.0)
            self.pt_resolution[pdgid] = (0.0, 0.0)
            self.eta_resolution[pdgid] = (0.0, 0.0)
            self.phi_resolution[pdgid] = (0.0, 0.0)
        self.pt_resolution["met"] = (0.0, 0.0)

        # Initialize samples
        self.reference_benchmark = None
        self.observations = None
        self.weights = None
        self.events_sampling_benchmark_ids = None

        # Initialize nuisance parameters
        self.nuisance_parameters = None

        # Initialize event summary
        self.signal_events_per_benchmark = None
        self.background_events = None

        # Information from .h5 file
        self.filename = filename
        (parameters, benchmarks, _, _, _, _, _, self.systematics, _, _, _, _) = load_madminer_settings(
            filename, include_nuisance_benchmarks=False
        )
        self.benchmark_names_phys = list(benchmarks.keys())
        self.n_benchmarks_phys = len(benchmarks)

    def add_sample(self, lhe_filename, sampled_from_benchmark, is_background=False, k_factor=1.0):
        """
        Adds an LHE sample of simulated events.

        Parameters
        ----------
        lhe_filename : str
            Path to the LHE event file (with extension '.lhe' or '.lhe.gz').

        sampled_from_benchmark : str
            Name of the benchmark that was used for sampling in this event file (the keyword `sample_benchmark`
            of `madminer.core.MadMiner.run()`).

        is_background : bool, optional
            Whether the sample is a background sample (i.e. without benchmark reweighting).

        k_factor : float, optional
            Multiplies the cross sections found in the sample. Default value: 1.

        Returns
        -------
            None

        """

        logger.debug("Adding event sample %s", lhe_filename)

        # Check inputs
        self.sampling_benchmarks.append(sampled_from_benchmark)
        self.sample_is_backgrounds.append(is_background)
        self.sample_k_factors.append(k_factor)
        self.lhe_sample_filenames.append(lhe_filename)

    def set_smearing(
        self,
        pdgids=None,
        energy_resolution_abs=0.0,
        energy_resolution_rel=0.0,
        pt_resolution_abs=0.0,
        pt_resolution_rel=0.0,
        eta_resolution_abs=0.0,
        eta_resolution_rel=0.0,
        phi_resolution_abs=0.0,
        phi_resolution_rel=0.0,
    ):
        """
        Sets up the smearing of measured momenta from shower and detector effects.

        This function can be called with pdgids=None, in which case the settinigs are used for all visible particles,
        or with pdgids set to a list of PDG ids representing particles, for instance [11, -11] for electrons (and
        positrons).

        For all particles of this type, and for the energy, pT, phi, and eta, the measurement error is drawn from a
        Gaussian with mean 0 and standard deviation given by `(X_resolution_abs + X * X_resolution_rel)`. Here `X` is
        the quantity (E, pT, phi, eta) of interest and X_resolution_abs and X_resolution_rel are the corresponding
        keywords. In the case of energy and pT, values smaller than 0  will lead to a re-drawing of the measurement
        error.

        Instead of such numerical values, either the energy or pT resolution (but not both!) may be set to None. In
        this case, this quantity is calculated from the mass of the particle and all other smeared particles. For
        instance, when pt_resolution_abs is None or pt_resolution_rel is None, for the given particles the energy,
        phi, and eta are smeared (according to their respective resolutions). Then the transverse momentum is calculated
        from the on-shell condition `p^2 = m^2`, or `pT = sqrt(E^2 - m^2) / cosh(eta)`. When this does not have a
        solution, the pT is set to zero. On the other hand, when energy_resolution_abs is None or energy_resolution_abs
        is None, for the given particles the pT, phi, and eta are smeared, and then the energy is calculated as
        `E = sqrt(pT * cosh(eta))^2 + m^2)`.

        Parameters
        ----------
        pdgids : None or list of int, optional
            Defines the particles these smearing functions affect. If None, all particles are affected. Note that if
            set_smearing() is called multiple times for a given particle, the earlier calls will be forgotten and only
            the last smearing function will take effect. Default value: None.

        energy_resolution_abs : float or None, optional
            Absolute measurement uncertainty for the energy in GeV. None means that the energy is not smeared directly,
            but calculated from the on-shell condition. Default value: 0.

        energy_resolution_rel : float or None, optional
            Relative measurement uncertainty for the energy. None means that the energy is not smeared directly, but
            calculated from the on-shell condition. Default value: 0.

        pt_resolution_abs : float or None, optional
            Absolute measurement uncertainty for the pT in GeV. None means that the pT is not smeared directly, but
            calculated from the on-shell condition. Default value: 0.

        pt_resolution_rel : float or None, optional
            Relative measurement uncertainty for the pT. None means that the pT is not smeared directly, but
            calculated from the on-shell condition. Default value: 0.

        eta_resolution_abs : float, optional
            Absolute measurement uncertainty for eta. Default value: 0.

        eta_resolution_rel : float, optional
            Relative measurement uncertainty for eta. Default value: 0.

        phi_resolution_abs : float, optional
            Absolute measurement uncertainty for phi. Default value: 0.

        phi_resolution_rel : float, optional
            Relative measurement uncertainty for phi. Default value: 0.

        Returns
        -------
            None

        """

        if pdgids is None:
            pdgids = get_elementary_pdg_ids()

        for pdgid in pdgids:
            self.energy_resolution[pdgid] = (energy_resolution_abs, energy_resolution_rel)
            self.pt_resolution[pdgid] = (pt_resolution_abs, pt_resolution_rel)
            self.eta_resolution[pdgid] = (eta_resolution_abs, eta_resolution_rel)
            self.phi_resolution[pdgid] = (phi_resolution_abs, phi_resolution_rel)

    def set_met_noise(self, abs_=0.0, rel=0.0):
        """
        Sets up additional noise in the MET variable from shower and detector effects.

        By default, the MET is calculated based on all reconstructed visible particles, including the effect of the
        smearing of these particles (set with `set_smearing()`). But often the MET is in fact more affected by
        additional soft activity than by mismeasurements of the hard particles. This function adds a Gaussian random
        to each of the x and y components of the MET vector. The Gaussian has mean 0 and standard deviation
        `abs + rel * HT`, where `HT` is the scalar sum of the pT of all particles in the process. Everything
        is given in GeV.

        Parameters
        ----------
        abs_ : float, optional
            Absolute contribution to MET noise. Default value: 0.

        rel : float, optional
            Relative (to HT) contribution to MET noise. Default value: 0.

        Returns
        -------
            None

        """

        self.pt_resolution["met"] = (abs_, rel)

    def add_observable(self, name, definition, required=False, default=None):
        """
        Adds an observable as a string that can be parsed by Python's `eval()` function.

        Parameters
        ----------
        name : str
            Name of the observable. Since this name will be used in `eval()` calls for cuts, this should not contain
            spaces or special characters.

        definition : str
            An expression that can be parsed by Python's `eval()` function. As objects, all particles can be
            used: `e`, `mu`, `tau`, `j`, `a`, `l`, `v` provide lists of electrons, muons, taus, jets, photons, leptons (
            electrons and muons combined), and neutrinos, in each case sorted by descending transverse momentum. `met` provides a
            missing ET object. `p` gives all particles in the same order as in the LHE file (i.e. in the same order as
            defined in the MadGraph process card). All these objects are instances of `MadMinerParticle`, which
            inherits from scikit-hep's [LorentzVector](http://scikit-hep.org/api/math.html#vector-classes). See the link
            for a documentation of their properties. In addition, `MadMinerParticle` have  properties `charge` and
            `pdg_id`, which return the charge in units of elementary charges (i.e. an electron has `e[0].charge = -1.`),
            and the PDG particle ID. For instance, `"abs(j[0].phi() - j[1].phi())"` defines the azimuthal angle between
            the two hardest jets.

        required : bool, optional
            Whether the observable is required. If True, an event will only be retained if this observable is
            successfully parsed. For instance, any observable involving `"j[1]"` will only be parsed if there are at
            least two jets passing the acceptance cuts. Default value: False.

        default : float or None, optional
            If `required=False`, this is the placeholder value for observables that cannot be parsed. None is replaced
            with `np.nan`. Default value: None.

        Returns
        -------
            None

        """

        if required:
            logger.debug("Adding required observable %s = %s", name, definition)
        else:
            logger.debug("Adding optional observable %s = %s with default %s", name, definition, default)

        self.observables[name] = definition
        self.observables_required[name] = required
        self.observables_defaults[name] = default

    def add_observable_from_function(self, name, fn, required=False, default=None):
        """
        Adds an observable defined through a function.

        Parameters
        ----------
        name : str
            Name of the observable. Since this name will be used in `eval()` calls for cuts, this should not contain
            spaces or special characters.

        fn : function
            A function with signature `observable(particles, leptons, photons, jets, met)` where all arguments are lists of
            MadMinerParticle instances and a float is returned. `particles` are the truth-level particles, ordered in the
            same way as in the LHE file, and no smearing is applied. `leptons`, `photons`, `jets`, and `met` have
            smearing applied. The function should raise a `RuntimeError` to signal that it is not defined.

        required : bool, optional
            Whether the observable is required. If True, an event will only be retained if this observable is
            successfully parsed. For instance, any observable involving `"j[1]"` will only be parsed if there are at
            least two jets passing the acceptance cuts. Default value: False.

        default : float or None, optional
            If `required=False`, this is the placeholder value for observables that cannot be parsed. None is replaced
            with `np.nan`. Default value: None.

        Returns
        -------
            None

        """

        if required:
            logger.debug("Adding required observable %s defined through external function", name)
        else:
            logger.debug(
                "Adding optional observable %s defined through external function with default %s", name, default
            )

        self.observables[name] = fn
        self.observables_required[name] = required
        self.observables_defaults[name] = default

    def add_default_observables(
        self,
        n_leptons_max=2,
        n_photons_max=2,
        n_jets_max=2,
        include_met=True,
        include_visible_sum=True,
        include_numbers=True,
        include_charge=True,
    ):
        """
        Adds a set of simple standard observables: the four-momenta (parameterized as E, pT, eta, phi) of the hardest
        visible particles, and the missing transverse energy.

        Parameters
        ----------
        n_leptons_max : int, optional
            Number of hardest leptons for which the four-momenta are saved. Default value: 2.

        n_photons_max : int, optional
            Number of hardest photons for which the four-momenta are saved. Default value: 2.

        n_jets_max : int, optional
            Number of hardest jets for which the four-momenta are saved. Default value: 2.

        include_met : bool, optional
            Whether the missing energy observables are stored. Default value: True.

        include_visible_sum : bool, optional
            Whether observables characterizing the sum of all particles are stored. Default value: True.

        include_numbers : bool, optional
            Whether the number of leptons, photons, and jets is saved as observable. Default value: True.

        include_charge : bool, optional
            Whether the lepton charge is saved as observable. Default value: True.

        Returns
        -------
            None

        """
        logger.debug("Adding default observables")

        # ETMiss
        if include_met:
            self.add_observable("et_miss", "met.pt", required=True)
            self.add_observable("phi_miss", "met.phi()", required=True)

        # Sum of visible particles
        if include_visible_sum:
            self.add_observable("e_visible", "visible.e", required=True)
            self.add_observable("eta_visible", "visible.eta", required=True)

        # Individual observed particles
        for n, symbol, include_this_charge in zip(
            [n_leptons_max, n_photons_max, n_jets_max], ["l", "a", "j"], [False, False, include_charge]
        ):
            if include_numbers:
                self.add_observable("n_{}s".format(symbol), "len({})".format(symbol), required=True)

            for i in range(n):
                self.add_observable(
                    "e_{}{}".format(symbol, i + 1), "{}[{}].e".format(symbol, i), required=False, default=0.0
                )
                self.add_observable(
                    "pt_{}{}".format(symbol, i + 1), "{}[{}].pt".format(symbol, i), required=False, default=0.0
                )
                self.add_observable(
                    "eta_{}{}".format(symbol, i + 1), "{}[{}].eta".format(symbol, i), required=False, default=0.0
                )
                self.add_observable(
                    "phi_{}{}".format(symbol, i + 1), "{}[{}].phi()".format(symbol, i), required=False, default=0.0
                )
                if include_this_charge and symbol == "l":
                    self.add_observable(
                        "charge_{}{}".format(symbol, i + 1),
                        "{}[{}].charge".format(symbol, i),
                        required=False,
                        default=0.0,
                    )

    def add_cut(self, definition, pass_if_not_parsed=False):

        """
        Adds a cut as a string that can be parsed by Python's `eval()` function and returns a bool.

        Parameters
        ----------
        definition : str
            An expression that can be parsed by Python's `eval()` function and returns a bool: True for the event
            to pass this cut, False for it to be rejected. In the definition, all visible particles can be
            used: `e`, `mu`, `j`, `a`, and `l` provide lists of electrons, muons, jets, photons, and leptons (electrons
            and muons combined), in each case sorted by descending transverse momentum. `met` provides a missing ET
            object. `visible` and `all` provide access to the sum of all visible particles and the sum of all visible
            particles plus MET, respectively. All these objects are instances of `MadMinerParticle`, which inherits from
            scikit-hep's [LorentzVector](http://scikit-hep.org/api/math.html#vector-classes). See the link for a
            documentation of their properties. In addition, `MadMinerParticle` have  properties `charge` and `pdg_id`,
            which return the charge in units of elementary charges (i.e. an electron has `e[0].charge = -1.`), and the
            PDG particle ID. For instance, `"len(e) >= 2"` requires at least two electrons passing the cuts,
            while `"mu[0].charge > 0."` specifies that the hardest muon is positively charged.

        pass_if_not_parsed : bool, optional
            Whether the cut is passed if the observable cannot be parsed. Default value: False.

        Returns
        -------
            None

        """
        logger.debug("Adding cut %s", definition)

        self.cuts.append(definition)
        self.cuts_default_pass.append(pass_if_not_parsed)

    def add_efficiency(self, definition, value_if_not_parsed=1.0):

        """
            Adds an efficiency as a string that can be parsed by Python's `eval()` function and returns a bool.

            Parameters
            ----------
            definition : str
            An expression that can be parsed by Python's `eval()` function and returns a floating number which reweights
            the event weights. In the definition, all visible particles can be used: `e`, `mu`, `j`, `a`, and `l` provide
            lists of electrons, muons, jets, photons, and leptons (electrons and muons combined), in each case sorted
            by descending transverse momentum. `met` provides a missing ET object. `visible` and `all` provide access to
            the sum of all visible particles and the sum of all visible particles plus MET, respectively. All these
            objects are instances of `MadMinerParticle`, which inherits from scikit-hep's
            [LorentzVector](http://scikit-hep.org/api/math.html#vector-classes). See the link for a
            documentation of their properties. In addition, `MadMinerParticle` have  properties `charge` and `pdg_id`,
            which return the charge in units of elementary charges (i.e. an electron has `e[0].charge = -1.`), and the
            PDG particle ID.

            value_if_not_parsed : float, optional
            Value if te efficiency function cannot be parsed. Default value: 1.

            Returns
            -------
            None

            """
        logger.debug("Adding efficiency %s", definition)

        self.efficiencies.append(definition)
        self.efficiencies_default_pass.append(value_if_not_parsed)

    def reset_observables(self):
        """ Resets all observables. """

        logger.debug("Resetting observables")

        self.observables = OrderedDict()
        self.observables_required = OrderedDict()
        self.observables_defaults = OrderedDict()

    def reset_cuts(self):
        """ Resets all cuts. """

        logger.debug("Resetting cuts")

        self.cuts = []
        self.cuts_default_pass = []

    def reset_efficiencies(self):
        """ Resets all efficiencies. """

        logger.debug("Resetting efficiencies")

        self.efficiencies = []
        self.efficiencies_default_pass = []

    def analyse_samples(self, reference_benchmark=None, parse_events_as_xml=True):
        """
        Main function that parses the LHE samples, applies detector effects, checks cuts,
        evaulate efficiencies, and extracts the observables and weights.

        Parameters
        ----------
        reference_benchmark : str or None, optional
            The weights at the nuisance benchmarks will be rescaled to some reference theta benchmark:
            `dsigma(x|theta_sampling(x),nu) -> dsigma(x|theta_ref,nu) = dsigma(x|theta_sampling(x),nu)
            * dsigma(x|theta_ref,0) / dsigma(x|theta_sampling(x),0)`. This sets the name of the reference benchmark.
            If None, the first one will be used. Default value: None.

        parse_events_as_xml : bool, optional
            Decides whether the LHE events are parsed with an XML parser (more robust, but slower) or a text parser
            (less robust, faster). Default value: True.

        Returns
        -------
            None

        """

        # Input
        if reference_benchmark is None:
            reference_benchmark = self.benchmark_names_phys[0]
        self.reference_benchmark = reference_benchmark

        # Reset observations
        self.observations = None
        self.weights = None
        self.nuisance_parameters = None
        self.events_sampling_benchmark_ids = None
        self.signal_events_per_benchmark = [0 for _ in range(self.n_benchmarks_phys)]
        self.background_events = 0

        for lhe_file, is_background, sampling_benchmark, k_factor in zip(
            self.lhe_sample_filenames, self.sample_is_backgrounds, self.sampling_benchmarks, self.sample_k_factors
        ):
            logger.info("Analysing LHE sample %s", lhe_file)

            this_observations, this_weights, this_n_events = self._parse_sample(
                is_background, k_factor, lhe_file, parse_events_as_xml, reference_benchmark, sampling_benchmark
            )

            # No results?
            if this_observations is None:
                continue

            # Store sampling id for each event
            if is_background:
                idx = -1
                self.background_events += this_n_events
            else:
                idx = self.benchmark_names_phys.index(sampling_benchmark)
                self.signal_events_per_benchmark[idx] += this_n_events
            this_events_sampling_benchmark_ids = np.array([idx] * this_n_events, dtype=np.int)

            # First results
            if self.observations is None and self.weights is None:
                self.observations = this_observations
                self.weights = this_weights
                self.events_sampling_benchmark_ids = this_events_sampling_benchmark_ids
                continue

            # Following results: check consistency with previous results
            if len(self.weights) != len(this_weights):
                raise ValueError(
                    "Number of weights in different files incompatible: {} vs {}".format(
                        len(self.weights), len(this_weights)
                    )
                )
            if len(self.observations) != len(this_observations):
                raise ValueError(
                    "Number of observations in different Delphes files incompatible: {} vs {}".format(
                        len(self.observations), len(this_observations)
                    )
                )

            # Merge results with previous
            for key in self.weights:
                assert key in this_weights, "Weight label {} not found in sample!".format(key)
                self.weights[key] = np.hstack([self.weights[key], this_weights[key]])

            for key in self.observations:
                assert key in this_observations, "Observable {} not found in Delphes sample!".format(key)
                self.observations[key] = np.hstack([self.observations[key], this_observations[key]])

            self.events_sampling_benchmark_ids = np.hstack(
                [self.events_sampling_benchmark_ids, this_events_sampling_benchmark_ids]
            )

        logger.info("Analysed number of events per sampling benchmark:")
        for name, n_events in zip(self.benchmark_names_phys, self.signal_events_per_benchmark):
            if n_events > 0:
                logger.info("  %s from %s", n_events, name)
        if self.background_events > 0:
            logger.info("  %s from backgrounds", self.background_events)

    def _parse_sample(
        self, is_background, k_factor, lhe_file, parse_events_as_xml, reference_benchmark, sampling_benchmark
    ):
        # Read systematics setup from LHE file
        logger.debug("Extracting nuisance parameter definitions from LHE file")
        nuisance_parameters = extract_nuisance_parameters_from_lhe_file(lhe_file, self.systematics)
        logger.debug("Found %s nuisance parameters with matching benchmarks:", len(nuisance_parameters))
        for key, value in six.iteritems(nuisance_parameters):
            logger.debug("  %s: %s", key, value)

        # Compare to existing data
        if self.nuisance_parameters is None:
            self.nuisance_parameters = nuisance_parameters
        else:
            if dict(self.nuisance_parameters) != dict(nuisance_parameters):
                raise RuntimeError(
                    "Different LHE files have different definitions of nuisance parameters / benchmarks!\nPrevious: {}\nNew:{}".format(
                        self.nuisance_parameters, nuisance_parameters
                    )
                )

        # Calculate observables and weights in LHE file
        this_observations, this_weights = parse_lhe_file(
            filename=lhe_file,
            sampling_benchmark=sampling_benchmark,
            benchmark_names=self.benchmark_names_phys,
            is_background=is_background,
            observables=self.observables,
            observables_required=self.observables_required,
            observables_defaults=self.observables_defaults,
            cuts=self.cuts,
            cuts_default_pass=self.cuts_default_pass,
            efficiencies=self.efficiencies,
            efficiencies_default_pass=self.efficiencies_default_pass,
            energy_resolutions=self.energy_resolution,
            pt_resolutions=self.pt_resolution,
            eta_resolutions=self.eta_resolution,
            phi_resolutions=self.phi_resolution,
            k_factor=k_factor,
            parse_events_as_xml=parse_events_as_xml,
        )

        # No events found?
        if this_observations is None:
            logger.debug("No observations in this LHE file, skipping it")
            return None, None
        logger.debug("Found weights %s in LHE file", list(this_weights.keys()))

        # Sanity checks
        n_events = self._check_sample_observations_and_weights(this_observations, this_weights)

        # Rescale nuisance parameters to reference benchmark
        reference_weights = this_weights[reference_benchmark]
        sampling_weights = this_weights[sampling_benchmark]
        for key in this_weights:
            if key not in self.benchmark_names_phys:  # Only rescale nuisance benchmarks
                this_weights[key] = reference_weights / sampling_weights * this_weights[key]

        return this_observations, this_weights, n_events

    @staticmethod
    def _check_sample_observations_and_weights(this_observations, this_weights):
        """ Sanity checks """

        # Check number of events in observables, and their dtype
        n_events = None
        for key, obs in six.iteritems(this_observations):
            this_n_events = len(obs)
            logger.debug("Found {} events in Obs {}".format(this_n_events, key))
            if n_events is None:
                n_events = this_n_events
                logger.debug("Found %s events", n_events)

            if this_n_events != n_events:
                raise RuntimeError(
                    "Mismatching number of events in LHE observations for {}: {} vs {}".format(
                        key, n_events, this_n_events
                    )
                )

            if not np.issubdtype(obs.dtype, np.number):
                logger.warning(
                    "Observations for observable %s have non-numeric dtype %s. This usually means something "
                    "is wrong in the definition of the observable. Data: %s",
                    key,
                    obs.dtype,
                    obs,
                )
        # Check number of events in weights, and thier dtype
        for key, weights in six.iteritems(this_weights):
            this_n_events = len(weights)
            if n_events is None:
                n_events = this_n_events
                logger.debug("Found %s events", n_events)

            if this_n_events != n_events:
                raise RuntimeError(
                    "Mismatching number of events in weights {}: {} vs {}".format(key, n_events, this_n_events)
                )

            if not np.issubdtype(weights.dtype, np.number):
                logger.warning(
                    "Weights %s have non-numeric dtype %s. This usually means something "
                    "is wrong in the definition of the observable. Data: %s",
                    key,
                    weights.dtype,
                    weights,
                )
        return n_events

    def save(self, filename_out):
        """
        Saves the observable definitions, observable values, and event weights in a MadMiner file. The parameter,
        benchmark, and morphing setup is copied from the file provided during initialization. Nuisance benchmarks found
        in the LHE file are added.

        Parameters
        ----------
        filename_out : str
            Path to where the results should be saved.

        Returns
        -------
            None

        """

        if self.observations is None or self.weights is None:
            logger.warning("No events to save!")
            return

        logger.debug("Loading HDF5 data from %s and saving file to %s", self.filename, filename_out)

        # Save nuisance parameters and benchmarks
        weight_names = list(self.weights.keys())
        logger.debug("Weight names: %s", weight_names)

        save_nuisance_setup_to_madminer_file(
            filename_out,
            weight_names,
            self.nuisance_parameters,
            reference_benchmark=self.reference_benchmark,
            copy_from=self.filename,
        )

        # Save events
        save_events_to_madminer_file(
            filename_out,
            self.observables,
            self.observations,
            self.weights,
            self.events_sampling_benchmark_ids,
            self.signal_events_per_benchmark,
            self.background_events,
        )
