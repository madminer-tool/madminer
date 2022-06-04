import logging
import numpy as np

from collections import OrderedDict
from pathlib import Path

from madminer.models import Cut
from madminer.models import Observable
from madminer.models import NuisanceParameter
from madminer.utils.interfaces.delphes import run_delphes
from madminer.utils.interfaces.delphes_root import parse_delphes_root_file
from madminer.utils.interfaces.hdf5 import load_madminer_settings
from madminer.utils.interfaces.hdf5 import save_events
from madminer.utils.interfaces.hdf5 import save_nuisance_setup
from madminer.utils.interfaces.hepmc import extract_weight_order
from madminer.utils.interfaces.lhe import parse_lhe_file, extract_nuisance_parameters_from_lhe_file
from madminer.sampling import combine_and_shuffle

logger = logging.getLogger(__name__)


class DelphesReader:
    """
    Detector simulation with Delphes and simple calculation of observables.

    After setting up the parameter space and benchmarks and running MadGraph and Pythia, all of which is organized
    in the madminer.core.MadMiner class, the next steps are the simulation of detector effects and the calculation of
    observables.  Different tools can be used for these tasks, please feel free to implement the detector simulation and
    analysis routine of your choice.

    This class provides an example implementation based on Delphes. Its workflow consists of the following steps:

    * Initializing the class with the filename of a MadMiner HDF5 file (the output of `madminer.core.MadMiner.save()`)
    * Adding one or multiple event samples produced by MadGraph and Pythia in `DelphesProcessor.add_sample()`.
    * Running Delphes on the samples that require it through `DelphesProcessor.run_delphes()`.
    * Optionally, acceptance cuts for all visible particles can be defined with `DelphesProcessor.set_acceptance()`.
    * Defining observables through `DelphesProcessor.add_observable()` or
      `DelphesProcessor.add_observable_from_function()`. A simple set of default observables is provided in
      `DelphesProcessor.add_default_observables()`
    * Optionally, cuts can be set with `DelphesProcessor.add_cut()`
    * Calculating the observables from the Delphes ROOT files with `DelphesProcessor.analyse_delphes_samples()`
    * Saving the results with `DelphesProcessor.save()`

    Please see the tutorial for a detailed walk-through.

    Parameters
    ----------
    filename : str or None, optional
        Path to MadMiner file (the output of `madminer.core.MadMiner.save()`). Default value: None.

    """

    def __init__(self, filename):
        # Initialize samples
        self.hepmc_sample_filenames = []
        self.hepmc_sample_weight_labels = []
        self.hepmc_sampled_from_benchmark = []
        self.hepmc_is_backgrounds = []
        self.lhe_sample_filenames = []
        self.lhe_sample_filenames_for_weights = []
        self.delphes_sample_filenames = []
        self.sample_k_factors = []
        self.sample_systematics = []

        # Initialize observables
        self.observables = OrderedDict()

        # Initialize cuts
        self.cuts = []

        # Initialize acceptance cuts
        self.acceptance_pt_min_e = None
        self.acceptance_pt_min_mu = None
        self.acceptance_pt_min_a = None
        self.acceptance_pt_min_j = None
        self.acceptance_eta_max_e = None
        self.acceptance_eta_max_mu = None
        self.acceptance_eta_max_a = None
        self.acceptance_eta_max_j = None

        # Initialize samples
        self.reference_benchmark = None
        self.observations = None
        self.weights = None
        self.events_sampling_benchmark_ids = []

        # Initialize event summary
        self.signal_events_per_benchmark = []
        self.background_events = 0

        # Information from .h5 file
        self.filename = filename

        (
            _,
            benchmarks,
            _,
            _,
            _,
            _,
            _,
            self.systematics,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = load_madminer_settings(filename, include_nuisance_benchmarks=False)

        self.benchmark_names_phys = list(benchmarks.keys())
        self.n_benchmarks_phys = len(benchmarks)

        # Initialize nuisance parameters
        self.nuisance_parameters = OrderedDict()

    @staticmethod
    def _check_sample_elements(this_elements, n_events=None):
        """ Sanity checks """

        # Check number of events in observables
        for key, elems in this_elements.items():
            this_n_events = len(elems)

            if n_events is None:
                n_events = this_n_events
                logger.debug(f"Found {n_events} events")

            if this_n_events != n_events:
                raise RuntimeError(f"Mismatching number of events for {key}: "f"{n_events} vs {this_n_events}")

            if not np.issubdtype(elems.dtype, np.number):
                logger.warning(f"For key {key} have non-numeric dtype {elems.dtype}.")

        return n_events

    def add_sample(
        self,
        hepmc_filename,
        sampled_from_benchmark,
        is_background=False,
        delphes_filename=None,
        lhe_filename=None,
        k_factor=1.0,
        weights="lhe",
        systematics=None,
    ):
        """
        Adds a sample of simulated events. A HepMC file (from Pythia) has to be provided always, since some relevant
        information is only stored in this file. The user can optionally provide a Delphes file, in this case
        run_delphes() does not have to be called.

        By default, the weights are read out from the Delphes file and their names from the HepMC file. There are some
        issues with current MadGraph versions that lead to Pythia not storing the weights. As work-around, MadMiner
        supports reading weights from the LHE file (the observables still come from the Delphes file). To enable this,
        use weights="lhe".

        Parameters
        ----------
        hepmc_filename : str
            Path to the HepMC event file (with extension '.hepmc' or '.hepmc.gz').

        sampled_from_benchmark : str
            Name of the benchmark that was used for sampling in this event file (the keyword `sample_benchmark`
            of `madminer.core.MadMiner.run()`).

        is_background : bool, optional
            Whether the sample is a background sample (i.e. without benchmark reweighting).

        delphes_filename : str or None, optional
            Path to the Delphes event file (with extension '.root'). If None, the user has to call run_delphes(), which
            will create this file. Default value: None.

        lhe_filename : None or str, optional
            Path to the LHE event file (with extension '.lhe' or '.lhe.gz'). This is only needed if weights is "lhe".

        k_factor : float, optional
            Multiplies the cross sections found in the sample. Default value: 1.

        weights : {"delphes", "lhe"}, optional
            If "delphes", the weights are read out from the Delphes ROOT file, and their names are taken from the
            HepMC file. If "lhe" (and lhe_filename is not None), the weights are taken from the LHE file (and matched
            with the observables from the Delphes ROOT file). The "delphes" behaviour is generally better as it
            minimizes the risk of mismatching observables and weights, but for some MadGraph and Delphes versions
            there are issues with weights not being saved in the HepMC and Delphes ROOT files. In this case, setting
            weights to "lhe" and providing the unweighted LHE file from MadGraph may be an easy fix. Default value:
            "lhe".

        systematics : None or list of str, optional
            List of systematics associated with this sample. Default value: None.

        Returns
        -------
            None

        """

        # Check inputs
        if hepmc_filename and not Path(hepmc_filename).exists():
            raise ValueError("The specified hepmc file does not exist")

        if lhe_filename and not Path(lhe_filename).exists():
            raise ValueError("The specified lhe file does not exist")

        if weights not in ["delphes", "lhe"]:
            raise ValueError("Unknown setting for weights. Has to be 'delphes' or 'lhe'.")

        if weights == "lhe" and lhe_filename is None:
            raise ValueError("With weights = 'lhe', a LHE event file has to be provided.")

        if self.systematics and lhe_filename is None:
            raise ValueError("With systematic uncertainties, a LHE event file has to be provided.")

        logger.debug("Adding event sample %s", hepmc_filename)

        self.hepmc_sample_filenames.append(hepmc_filename)
        self.hepmc_sampled_from_benchmark.append(sampled_from_benchmark)
        self.hepmc_is_backgrounds.append(is_background)
        self.sample_k_factors.append(k_factor)
        self.delphes_sample_filenames.append(delphes_filename)
        self.lhe_sample_filenames.append(lhe_filename)
        self.sample_systematics.append(systematics)

        if weights == "lhe" and lhe_filename is not None:
            self.hepmc_sample_weight_labels.append(None)
            self.lhe_sample_filenames_for_weights.append(lhe_filename)
        else:
            self.hepmc_sample_weight_labels.append(extract_weight_order(hepmc_filename, sampled_from_benchmark))
            self.lhe_sample_filenames_for_weights.append(None)

    def run_delphes(self, delphes_directory, delphes_card, initial_command=None, log_file=None):
        """
        Runs the fast detector simulation Delphes on all HepMC samples added so far for which it hasn't been run yet.

        Parameters
        ----------
        delphes_directory : str
            Path to the Delphes directory.

        delphes_card : str
            Path to a Delphes card.

        initial_command : str or None, optional
            Initial bash commands that have to be executed before Delphes is run (e.g. to load the correct virtual
            environment). Default value: None.

        log_file : str or None, optional
            Path to log file in which the Delphes output is saved. Default value: None.

        Returns
        -------
            None

        """

        if log_file is None:
            log_file = "./logs/delphes.log"

        for i, (delphes_filename, hepmc_filename) in enumerate(
            zip(self.delphes_sample_filenames, self.hepmc_sample_filenames)
        ):
            if delphes_filename is not None and Path(delphes_filename).is_file():
                logger.debug("Delphes already run for event sample %s", hepmc_filename)
                continue
            elif delphes_filename is not None:
                logger.debug(
                    "Given Delphes file %s does not exist, running Delphes again on HepMC sample at %s",
                    delphes_filename,
                    hepmc_filename,
                )
            else:
                logger.info("Running Delphes on HepMC sample at %s", hepmc_filename)

            delphes_sample_filename = run_delphes(
                delphes_directory=delphes_directory,
                delphes_card_filename=delphes_card,
                hepmc_sample_filename=hepmc_filename,
                initial_command=initial_command,
                log_file=log_file,
            )

            self.delphes_sample_filenames[i] = delphes_sample_filename

    def set_acceptance(
        self,
        pt_min_e=None,
        pt_min_mu=None,
        pt_min_a=None,
        pt_min_j=None,
        eta_max_e=None,
        eta_max_mu=None,
        eta_max_a=None,
        eta_max_j=None,
    ):
        """
        Sets acceptance cuts for all visible particles. These are taken into account before observables and cuts
        are calculated.

        Parameters
        ----------
        pt_min_e : float or None, optional
             Minimum electron transverse momentum in GeV. None means no acceptance cut. Default value: None.

        pt_min_mu : float or None, optional
             Minimum muon transverse momentum in GeV. None means no acceptance cut. Default value: None.

        pt_min_a : float or None, optional
             Minimum photon transverse momentum in GeV. None means no acceptance cut. Default value: None.

        pt_min_j : float or None, optional
             Minimum jet transverse momentum in GeV. None means no acceptance cut. Default value: None.

        eta_max_e : float or None, optional
             Maximum absolute electron pseudorapidity. None means no acceptance cut. Default value: None.

        eta_max_mu : float or None, optional
             Maximum absolute muon pseudorapidity. None means no acceptance cut. Default value: None.

        eta_max_a : float or None, optional
             Maximum absolute photon pseudorapidity. None means no acceptance cut. Default value: None.

        eta_max_j : float or None, optional
             Maximum absolute jet pseudorapidity. None means no acceptance cut. Default value: None.

        Returns
        -------
            None

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
        Adds an observable as a string that can be parsed by Python's `eval()` function.

        Parameters
        ----------
        name : str
            Name of the observable. Since this name will be used in `eval()` calls for cuts, this should not contain
            spaces or special characters.

        definition : str
            An expression that can be parsed by Python's `eval()` function. As objects, the visible particles can be
            used: `e`, `mu`, `j`, `a`, and `l` provide lists of electrons, muons, jets, photons, and leptons (electrons
            and muons combined), in each case sorted by descending transverse momentum. `met` provides a missing ET
            object. `visible` and `all` provide access to the sum of all visible particles and the sum of all visible
            particles plus MET, respectively. In addition, `MadMinerParticle` have  properties `charge` and `pdg_id`,
            which return the charge in units of elementary charges (i.e. an electron has `e[0].charge = -1.`), and the
            PDG particle ID. For instance, `"abs(j[0].phi - j[1].phi)"` defines the azimuthal angle between the two
            hardest jets.

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

        self.observables[name] = Observable(
            name=name,
            val_expression=definition,
            val_default=default,
            is_required=required,
        )

    def add_observable_from_function(self, name, fn, required=False, default=None):
        """
        Adds an observable defined through a function.

        Parameters
        ----------
        name : str
            Name of the observable. Since this name will be used in `eval()` calls for cuts, this should not contain
            spaces or special characters.

        fn : function
            A function with signature `observable(leptons, photons, jets, met)` where the input arguments are lists of
            MadMinerParticle instances and a float is returned. The function should raise a `RuntimeError` to signal
            that it is not defined.

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

        self.observables[name] = Observable(
            name=name,
            val_expression=fn,
            val_default=default,
            is_required=required,
        )

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
            self.add_observable("phi_miss", "met.phi", required=True)

        # Sum of visible particles
        if include_visible_sum:
            self.add_observable("e_visible", "visible.e", required=True)
            self.add_observable("eta_visible", "visible.eta", required=True)

        # Individual observed particles
        for n, symbol, include_this_charge in zip(
            [n_leptons_max, n_photons_max, n_jets_max], ["l", "a", "j"], [False, False, include_charge]
        ):
            if include_numbers:
                self.add_observable(f"n_{symbol}s", f"len({symbol})", required=True)

            for i in range(n):
                self.add_observable(
                    f"e_{symbol}{i+1}", f"{symbol}[{i}].e", required=False, default=0.0
                )
                self.add_observable(
                    f"pt_{symbol}{i+1}", f"{symbol}[{i}].pt", required=False, default=0.0
                )
                self.add_observable(
                    f"eta_{symbol}{i+1}", f"{symbol}[{i}].eta", required=False, default=0.0
                )
                self.add_observable(
                    f"phi_{symbol}{i+1}", f"{symbol}[{i}].phi", required=False, default=0.0
                )
                if include_this_charge and symbol == "l":
                    self.add_observable(
                        f"charge_{symbol}{i+1}",
                        f"{symbol}[{i}].charge",
                        required=False,
                        default=0.0,
                    )

    def add_cut(self, definition, required=False):
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
            particles plus MET, respectively. In addition, `MadMinerParticle` have  properties `charge` and `pdg_id`,
            which return the charge in units of elementary charges (i.e. an electron has `e[0].charge = -1.`), and the
            PDG particle ID. For instance, `"len(e) >= 2"` requires at least two electrons passing the acceptance cuts,
            while `"mu[0].charge > 0."` specifies that the hardest muon is positively charged.

        required : bool, optional
            Whether the cut is passed if the observable cannot be parsed. Default value: False.

        Returns
        -------
            None
        """

        logger.debug("Adding cut %s", definition)

        self.cuts.append(Cut(
            name="CUT",
            val_expression=definition,
            is_required=required,
        ))

    def reset_observables(self):
        """ Resets all observables. """

        logger.debug("Resetting observables")
        self.observables = OrderedDict()

    def reset_cuts(self):
        """ Resets all cuts. """

        logger.debug("Resetting cuts")
        self.cuts = []

    def analyse_delphes_samples(
        self,
        generator_truth=False,
        delete_delphes_files=False,
        reference_benchmark=None,
        parse_lhe_events_as_xml=True,
    ):
        """
        Main function that parses the Delphes samples (ROOT files), checks acceptance and cuts, and extracts
        the observables and weights.

        Parameters
        ----------
        generator_truth : bool, optional
            If True, the generator truth information (as given out by Pythia) will be parsed. Detector resolution or
            efficiency effects will not be taken into account.

        delete_delphes_files : bool, optional
            If True, the Delphes ROOT files will be deleted after extracting the information from them. Default value:
            False.

        reference_benchmark : str or None, optional
            The weights at the nuisance benchmarks will be rescaled to some reference theta benchmark:
            `dsigma(x|theta_sampling(x),nu) -> dsigma(x|theta_ref,nu) = dsigma(x|theta_sampling(x),nu)
            * dsigma(x|theta_ref,0) / dsigma(x|theta_sampling(x),0)`. This sets the name of the reference benchmark.
            If None, the first one will be used. Default value: None.

        parse_lhe_events_as_xml : bool, optional
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
        self.nuisance_parameters = OrderedDict()
        self.events_sampling_benchmark_ids = []
        self.signal_events_per_benchmark = [0 for _ in range(self.n_benchmarks_phys)]
        self.background_events = 0

        for (
            delphes_file,
            weight_labels,
            is_background,
            sampling_benchmark,
            lhe_file,
            lhe_file_for_weights,
            k_factor,
            sample_syst_names,
        ) in zip(
            self.delphes_sample_filenames,
            self.hepmc_sample_weight_labels,
            self.hepmc_is_backgrounds,
            self.hepmc_sampled_from_benchmark,
            self.lhe_sample_filenames,
            self.lhe_sample_filenames_for_weights,
            self.sample_k_factors,
            self.sample_systematics,
        ):
            logger.info(
                "Analysing Delphes sample %s: Calculating %s observables, requiring %s selection cuts, associated with "
                "%s",
                delphes_file,
                len(self.observables),
                len(self.cuts),
                "no systematics" if sample_syst_names is None else "systematics" + ", ".join(list(sample_syst_names)),
            )

            this_observations, this_weights, this_n_events = self._analyse_delphes_sample(
                delete_delphes_files,
                delphes_file,
                generator_truth,
                is_background,
                k_factor,
                lhe_file,
                lhe_file_for_weights,
                parse_lhe_events_as_xml,
                reference_benchmark,
                sampling_benchmark,
                weight_labels,
                sample_syst_names,
            )

            # No events?
            if this_observations is None:
                continue

            # Store sampling id for each event
            if is_background:
                idx = -1
                self.background_events += this_n_events
            else:
                idx = self.benchmark_names_phys.index(sampling_benchmark)
                self.signal_events_per_benchmark[idx] += this_n_events
            this_events_sampling_benchmark_ids = np.array([idx] * this_n_events, dtype=int)

            # First results
            if self.observations is None and self.weights is None:
                self.observations = this_observations
                self.weights = this_weights
                self.events_sampling_benchmark_ids = this_events_sampling_benchmark_ids
                continue

            # Following results: check consistency with previous results
            if len(self.observations) != len(this_observations):
                raise ValueError(
                    f"Number of observations in different Delphes files incompatible: "
                    f"{len(self.observations)} vs {len(this_observations)}"
                )

            # Merge weights with previous
            logging.debug("Merging data extracted from this file with data from previous files")
            previous_reference_weights = np.copy(self.weights[reference_benchmark])
            for key in self.weights:
                if key in this_weights:
                    # Benchmark exists in both samples
                    self.weights[key] = np.hstack([self.weights[key], this_weights[key]])
                    logging.debug("  Weights for benchmark %s exist in both", key)
                else:
                    # Benchmark only in previous samples
                    self.weights[key] = np.hstack([self.weights[key], this_weights[reference_benchmark]])
                    logging.debug("  Weights for benchmark %s exist only in previous files", key)
            for key in this_weights:
                if key in self.weights:
                    continue
                # Benchmark only in new samples
                self.weights[key] = np.hstack([previous_reference_weights, this_weights[key]])
                logging.debug("  Weights for benchmark %s exist only in new file", key)

            # Merge observations with previous (should always be the same observables)
            for key in self.observations:
                assert key in this_observations, f"Observable {key} not found in Delphes sample!"
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

    def _analyse_delphes_sample(
        self,
        delete_delphes_files,
        delphes_file,
        generator_truth,
        is_background,
        k_factor,
        lhe_file,
        lhe_file_for_weights,
        parse_lhe_events_as_xml,
        reference_benchmark,
        sampling_benchmark,
        weight_labels,
        sample_syst_names,
    ):
        # Relevant systematics
        systematics_used = OrderedDict()
        if sample_syst_names is None:
            sample_syst_names = []
        for key in sample_syst_names:
            systematics_used[key] = self.systematics[key]

        if len(systematics_used) > 0 and lhe_file_for_weights is None:
            raise NotImplementedError(
                "Systematic uncertainties are currently only supported when the weights"
                " are extracted from the LHE file (instead of the HepMC / Delphes ROOT"
                " file). Please use the keyword lhe_filename when calling add_sample()."
            )

        # Read systematics setup from LHE file
        logger.debug("Extracting nuisance parameter definitions from LHE file")
        systematics_dict = extract_nuisance_parameters_from_lhe_file(lhe_file, systematics_used)
        logger.debug("systematics_dict: %s", systematics_dict)

        # systematics_dict has structure
        # {systematics_name : {nuisance_parameter_name : ((benchmark0, weight0), (benchmark1, weight1), processing)}}

        # Store nuisance parameters
        for systematics_name, nuisance_info in systematics_dict.items():
            for nuisance_param_name, ((benchmark0, weight0), (benchmark1, weight1), _) in nuisance_info.items():
                nuisance_param = self.nuisance_parameters.get(nuisance_param_name)

                if nuisance_param is None:
                    raise RuntimeError(f"Nuisance parameter {nuisance_param_name} does not exist")
                if (
                    nuisance_param.systematic != systematics_name
                    or nuisance_param.benchmark_pos != benchmark0
                    or nuisance_param.benchmark_neg != benchmark1
                ):
                    raise RuntimeError(
                        f"Inconsistent information for same nuisance parameter {nuisance_param_name}. "
                        f"Old: {nuisance_param}. "
                        f"New: {(systematics_name, benchmark0, benchmark1)}."
                    )

                self.nuisance_parameters[nuisance_param_name] = NuisanceParameter(
                    name=nuisance_param_name,
                    systematic=systematics_name,
                    benchmark_pos=benchmark0,
                    benchmark_neg=benchmark1,
                )

        # Calculate observables and weights in Delphes ROOT file
        this_observations, this_weights, cut_filter = parse_delphes_root_file(
            delphes_file,
            self.observables,
            self.cuts,
            weight_labels,
            use_generator_truth=generator_truth,
            delete_delphes_sample_file=delete_delphes_files,
            acceptance_eta_max_a=self.acceptance_eta_max_a,
            acceptance_eta_max_e=self.acceptance_eta_max_e,
            acceptance_eta_max_mu=self.acceptance_eta_max_mu,
            acceptance_eta_max_j=self.acceptance_eta_max_j,
            acceptance_pt_min_a=self.acceptance_pt_min_a,
            acceptance_pt_min_e=self.acceptance_pt_min_e,
            acceptance_pt_min_mu=self.acceptance_pt_min_mu,
            acceptance_pt_min_j=self.acceptance_pt_min_j,
        )
        # No events found?
        if this_observations is None:
            logger.warning("No remaining events in this Delphes file, skipping it")
            return None, None, None

        if this_weights is not None:
            logger.debug("Found weights %s in Delphes file", list(this_weights.keys()))
        else:
            logger.debug("Did not extract weights from Delphes file")

        # Sanity checks
        n_events = self._check_sample_elements(this_observations, None)

        # Find weights in LHE file
        if lhe_file_for_weights is not None:
            logger.debug("Extracting weights from LHE file")
            _, this_weights = parse_lhe_file(
                filename=lhe_file_for_weights,
                sampling_benchmark=sampling_benchmark,
                observables=OrderedDict(),
                benchmark_names=self.benchmark_names_phys,
                is_background=is_background,
                parse_events_as_xml=parse_lhe_events_as_xml,
                systematics_dict=systematics_dict,
            )

            logger.debug("Found weights %s in LHE file", list(this_weights.keys()))

            # Apply cuts
            logger.debug("Applying Delphes-based cuts to LHE weights")
            for key, weights in this_weights.items():
                this_weights[key] = weights[cut_filter]

        if this_weights is None:
            raise RuntimeError("Could not extract weights from Delphes ROOT file or LHE file.")

        # Sanity checks
        n_events = self._check_sample_elements(this_weights, n_events)

        # k factors
        if k_factor is not None:
            for key in this_weights:
                this_weights[key] = k_factor * this_weights[key]

        # Background scenario: we only have one set of weights, but these should be true for all benchmarks
        if is_background:
            logger.debug("Sample is background")
            benchmarks_weight = list(this_weights.values())[0]

            for benchmark_name in self.benchmark_names_phys:
                this_weights[benchmark_name] = benchmarks_weight

        # Rescale nuisance parameters to reference benchmark
        reference_weights = this_weights[reference_benchmark]
        sampling_weights = this_weights[sampling_benchmark]
        for key in this_weights:
            if key not in self.benchmark_names_phys:  # Only rescale nuisance benchmarks
                this_weights[key] = reference_weights / sampling_weights * this_weights[key]

        return this_observations, this_weights, n_events

    def save(self, filename_out, shuffle=True):
        """
        Saves the observable definitions, observable values, and event weights in a MadMiner file. The parameter,
        benchmark, and morphing setup is copied from the file provided during initialization. Nuisance benchmarks found
        in the HepMC file are added.

        Parameters
        ----------
        filename_out : str
            Path to where the results should be saved.

        shuffle : bool, optional
            If True, events are shuffled before being saved. That's important when there are multiple distinct
            samples (e.g. signal and background). Default value: True.

        Returns
        -------
            None

        """

        if self.observations is None or self.weights is None:
            logger.warning("No observations to save!")
            return

        logger.debug("Loading HDF5 data from %s and saving file to %s", self.filename, filename_out)

        # Save nuisance parameters and benchmarks
        weight_names = list(self.weights.keys())
        logger.debug("Weight names: %s", weight_names)

        save_nuisance_setup(
            file_name=filename_out,
            file_override=True,
            nuisance_benchmarks=weight_names,
            nuisance_parameters=self.nuisance_parameters,
            reference_benchmark=self.reference_benchmark,
            copy_from_path=self.filename,
        )

        # Save events
        save_events(
            file_name=filename_out,
            file_override=True,
            observables=self.observables,
            observations=self.observations,
            weights=self.weights,
            sampling_benchmarks=self.events_sampling_benchmark_ids,
            num_signal_events=self.signal_events_per_benchmark,
            num_background_events=self.background_events,
        )

        if shuffle:
            combine_and_shuffle([filename_out], filename_out)
