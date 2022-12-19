import os
import logging
import tempfile

from collections import OrderedDict
from pathlib import Path
from typing import Dict
from typing import List
from typing import Union

from madminer.models import AnalysisParameter
from madminer.models import Benchmark
from madminer.models import Systematic
from madminer.models import SystematicScale
from madminer.models import SystematicType
from madminer.utils.morphing import PhysicsMorpher
from madminer.utils.interfaces.hdf5 import load_madminer_settings
from madminer.utils.interfaces.hdf5 import save_madminer_settings
from madminer.utils.interfaces.mg_cards import export_param_card
from madminer.utils.interfaces.mg_cards import export_reweight_card
from madminer.utils.interfaces.mg_cards import export_run_card
from madminer.utils.interfaces.mg import generate_mg_process
from madminer.utils.interfaces.mg import setup_mg_with_scripts
from madminer.utils.interfaces.mg import run_mg
from madminer.utils.interfaces.mg import create_master_script
from madminer.utils.interfaces.mg import setup_mg_reweighting_with_scripts
from madminer.utils.interfaces.mg import run_mg_reweighting
from madminer.utils.various import copy_file

logger = logging.getLogger(__name__)


class MadMiner:
    """
    The central class to manage parameter spaces, benchmarks, and the generation of events through MadGraph and
    Pythia.

    An instance of this class is the starting point of most MadMiner applications. It is typically used in four steps:

    * Defining the parameter space through `MadMiner.add_parameter`
    * Defining the benchmarks, i.e. the points at which the squared matrix elements will be evaluated in MadGraph, with
      `MadMiner.add_benchmark()` or, if operator morphing is used, with `MadMiner.set_benchmarks_from_morphing()`
    * Saving this setup with `MadMiner.save()` (it can be loaded in a new instance with `MadMiner.load()`)
    * Running MadGraph and Pythia with the appropriate settings with `MadMiner.run()` or `MadMiner.run_multiple()` (the
      latter allows the user to combine runs from multiple run cards and sampling points)

    Please see the tutorial for a hands-on introduction to its methods.

    """

    def __init__(self):
        self.parameters = OrderedDict()
        self.benchmarks = OrderedDict()
        self.default_benchmark = None
        self.morpher = None
        self.export_morphing = False
        self.systematics = OrderedDict()
        self.finite_difference_benchmarks = OrderedDict()
        self.finite_difference_epsilon = 0.0

    def _reset_systematics(self):
        self.systematics = OrderedDict()

    def _reset_benchmarks(self):
        self.benchmarks = OrderedDict()
        self.default_benchmark = None

    def _reset_morpher(self):
        self.morpher = None
        self.export_morphing = False

    def add_parameter(
        self,
        lha_block,
        lha_id,
        parameter_name=None,
        param_card_transform=None,
        morphing_max_power=2,
        parameter_range=(0.0, 1.0),
    ):
        """
        Adds an individual parameter.

        Parameters
        ----------
        lha_block :  str
            The name of the LHA block as used in the param_card. Case-sensitive.

        lha_id : int
            The LHA id as used in the param_card.

        parameter_name : str or None
            An internal name for the parameter. If None, a the default 'benchmark_i' is used.

        morphing_max_power : int
            The maximal power with which this parameter contributes to the
            squared matrix element of the process of interest. Typically at tree level,
            this maximal number is 2 for parameters that affect one vertex (e.g. only production
            or only decay of a particle), and 4 for parameters that affect two vertices (e.g.
            production and decay). Default value: 2.

        param_card_transform : None or str
            Represents a one-parameter function mapping the parameter
            (`"theta"`) to the value that should be written in the parameter cards. This
            str is parsed by Python's `eval()` function, and `"theta"` is parsed as the
            parameter value. Default value: None.

        parameter_range : tuple of float
            The range of parameter values of primary interest. Only affects the
            basis optimization. Default value: (0., 1.).

        Returns
        -------
            None
        """

        # Default names
        if parameter_name is None:
            parameter_name = f"parameter_{len(self.parameters)}"
        if param_card_transform is None:
            param_card_transform = "_"

        # Check and sanitize input
        assert isinstance(lha_block, str), f"LHA block is not a string: {lha_block}"
        assert isinstance(lha_id, int), f"LHA id is not an integer: {lha_id}"
        assert isinstance(parameter_name, str), f"Parameter name is not a string: {parameter_name}"
        assert isinstance(morphing_max_power, int), f"Morphing max power is not an integer: {morphing_max_power}"

        parameter_name = parameter_name.replace(" ", "_")
        parameter_name = parameter_name.replace("-", "_")

        assert parameter_name not in self.parameters, f"Parameter already exists: {parameter_name}"

        parameter = AnalysisParameter(
            parameter_name,
            lha_block,
            lha_id,
            morphing_max_power,
            parameter_range,
            param_card_transform,
        )

        # Add parameter
        logger.info("Adding parameter: %s", parameter)
        self.parameters[parameter_name] = parameter

        # The morphing information is not accurate anymore
        logger.warning("Resetting benchmarks and morphing")
        self._reset_benchmarks()
        self._reset_morpher()

    def set_parameters(self, parameters: Union[Dict[str, AnalysisParameter], List[tuple]]):
        """
        Manually sets all parameters, overwriting previously added parameters.

        Parameters
        ----------
        parameters : dict or list
             If parameters is an dict, the keys should be str and give the parameter names, and the values are
             AnalysisParameter model instances. If parameters is a list, the items should be tuples of the
             form (LHA_block, LHA_ID).

        Returns
        -------
            None
        """

        self.parameters = OrderedDict()

        if isinstance(parameters, dict):
            for param in parameters.values():
                self.add_parameter(
                    lha_block=param.lha_block,
                    lha_id=param.lha_id,
                    parameter_name=param.name,
                    morphing_max_power=param.max_power,
                    parameter_range=param.val_range,
                )
        elif isinstance(parameters, list):
            for values in parameters:
                self.add_parameter(values[0], values[1])
        else:
            raise RuntimeError(f"Invalid set of parameters: {parameters}")

        # The morphing information is not accurate anymore
        logger.warning("Resetting benchmarks and morphing")
        self._reset_benchmarks()
        self._reset_morpher()

    def add_benchmark(self, parameter_values: Dict[str, float], benchmark_name: str = None, verbose: float = True):
        """
        Manually adds an individual benchmark, that is, a parameter point that will be evaluated by MadGraph.

        Parameters
        ----------
        parameter_values : dict
            The keys of this dict should be the parameter names and the values the corresponding parameter values.
        benchmark_name : str or None, optional
            Name of benchmark. If None, a default name is used. Default value: None.
        verbose : bool, optional
            If True, prints output about each benchmark. Default value: True.

        Returns
        -------
            None

        Raises
        ------
        RuntimeError
            If a benchmark with the same name already exists, if parameter_values is not a dict, or if a key of
            parameter_values does not correspond to a defined parameter.
        """

        # Default names
        if benchmark_name is None:
            benchmark_name = f"benchmark_{len(self.benchmarks)}"

        # Check input
        if not isinstance(parameter_values, dict):
            raise RuntimeError(f"Parameter values are not a dict: {parameter_values}")

        for p_name in parameter_values.keys():
            if p_name not in self.parameters.keys():
                raise RuntimeError(f"Unknown parameter: {p_name}")

        if benchmark_name in self.benchmarks.keys():
            raise RuntimeError(f"Benchmark {benchmark_name} exists already")

        # Add benchmark
        self.benchmarks[benchmark_name] = Benchmark(
            name=benchmark_name,
            values=parameter_values,
        )

        # If first benchmark, this will be the default for sampling
        if len(self.benchmarks) == 1:
            self.default_benchmark = benchmark_name

        if verbose:
            logger.info("Added benchmark %s", self.benchmarks[benchmark_name])
        else:
            logger.debug("Added benchmark %s", self.benchmarks[benchmark_name])

    def set_benchmarks(self, benchmarks: Union[Dict[str, dict], List[dict]], verbose: bool = True):
        """
        Manually sets all benchmarks, that is, parameter points that will be evaluated by MadGraph. Calling this
        function overwrites all previously defined benchmarks.

        Parameters
        ----------
        benchmarks : dict or list
            Specifies all benchmarks. If None, all benchmarks are reset. If dict, the keys are the benchmark names and
            the values the Benchmark instances. If list, the entries are dicts {parameter_name:value}
            (and the benchmark names are chosen automatically). Default value: None.

        verbose : bool, optional
            If True, prints output about each benchmark. Default value: True.

        Returns
        -------
            None
        """

        self.benchmarks = OrderedDict()
        self.default_benchmark = None

        if isinstance(benchmarks, dict):
            for name, values in benchmarks.items():
                self.add_benchmark(values, name, verbose=verbose)
        elif isinstance(benchmarks, list):
            for values in benchmarks:
                self.add_benchmark(values)
        else:
            raise RuntimeError(f"Invalid set of benchmarks: {benchmarks}")

        # After manually adding benchmarks, the morphing information is not accurate anymore
        if self.morpher is not None:
            logger.warning("Reset morphing")
            self.morpher = None
            self.export_morphing = False

    def set_morphing(
        self,
        max_overall_power=4,
        n_bases=1,
        include_existing_benchmarks=True,
        n_trials=100,
        n_test_thetas=100,
    ):
        """
        Sets up the morphing environment.

        Sets benchmarks, i.e. parameter points that will be evaluated by MadGraph, for a morphing algorithm, and
        calculates all information required for morphing. Morphing is a technique that allows MadMax to infer the full
        probability distribution `p(x_i | theta)` for each simulated event `x_i` and any `theta`, not just the
        benchmarks.

        The morphing basis is optimized with respect to the expected mean squared morphing weights over the parameter
        region of interest. If keep_existing_benchmarks=True, benchmarks defined previously will be incorporated in the
        morphing basis and only the remaining basis points will be optimized.

        Note that any subsequent call to `set_benchmarks` or `add_benchmark` will overwrite the morphing setup. The
        correct order is therefore to manually define benchmarks first, using `set_benchmarks` or `add_benchmark`, and
        then to create the morphing setup and complete the basis by calling
        `set_benchmarks_from_morphing(keep_existing_benchmarks=True)`.

        Parameters
        ----------
        max_overall_power : int, optional
            The maximal sum of powers of all parameters contributing to the squared matrix element.
            Typically, if parameters can affect the couplings at n vertices, this number is 2n. Default value: 4.

        n_bases : int, optional
            The number of morphing bases generated. If n_bases > 1, multiple bases are combined, and the
            weights for each basis are reduced by a factor 1 / n_bases. Currently only the default choice of 1 is
            fully implemented. Do not use any other value for now. Default value: 1.

        include_existing_benchmarks : bool, optional
            If True, the previously defined benchmarks are included in the morphing basis. In that case, the number of
            free parameters in the optimization routine is reduced. If False, the existing benchmarks will still be
            simulated, but are not part of the morphing routine. Default value: True.

        n_trials : int, optional
            Number of random basis configurations tested in the optimization procedure. A larger number will increase
            the run time of the optimization, but lead to better results. Default value: 100.

        n_test_thetas : int, optional
            Number of random parameter points used to evaluate the expected mean squared morphing weights. A larger
            number will increase the run time of the optimization, but lead to better results. Default value: 100.

        Returns
        -------
            None

        """

        logger.info("Optimizing basis for morphing")

        morpher = PhysicsMorpher(parameters_from_madminer=self.parameters)
        morpher.find_components(max_overall_power)

        if include_existing_benchmarks:
            n_predefined_benchmarks = len(self.benchmarks)
            basis = morpher.optimize_basis(
                n_bases=n_bases,
                benchmarks_from_madminer=self.benchmarks,
                n_trials=n_trials,
                n_test_thetas=n_test_thetas,
            )
        else:
            n_predefined_benchmarks = 0
            basis = morpher.optimize_basis(
                n_bases=n_bases,
                benchmarks_from_madminer=None,
                n_trials=n_trials,
                n_test_thetas=n_test_thetas,
            )

            basis.update(self.benchmarks)

        self.set_benchmarks(basis, verbose=False)
        self.morpher = morpher
        self.export_morphing = True

        logger.info(
            "Set up morphing with %s parameters, %s morphing components, %s predefined basis points, and %s "
            "new basis points",
            morpher.n_parameters,
            morpher.n_components,
            n_predefined_benchmarks,
            morpher.n_components - n_predefined_benchmarks,
        )

    def finite_differences(self, epsilon=0.01):
        """
        Adds benchmarks so that the score can be computed from finite differences

        Don't add any more benchmarks or parameters after calling this!
        """

        logger.info("Adding finite-differences benchmarks with epsilon = %s", epsilon)

        self.finite_difference_epsilon = epsilon

        # Copy is necessary to avoid endless loop :/
        for b_name, benchmark in self.benchmarks.copy().items():
            fd_keys = {}

            for param_name, param_value in benchmark.values.items():
                fd_key = f"{b_name}_plus_{param_name}"
                fd_obj = benchmark.copy()
                fd_obj.values[param_name] += epsilon

                self.add_benchmark(fd_obj, fd_key)
                fd_keys[param_name] = fd_key

            self.finite_difference_benchmarks[b_name].shift_names = fd_keys

    def add_systematics(
        self,
        effect,
        systematic_name=None,
        norm_variation=1.1,
        scale="mu",
        scale_variations=(0.5, 1.0, 2.0),
        pdf_variation="CT10",
    ):
        """

        Parameters
        ----------
        effect : {"norm", "scale", "pdf"}
            Type of the nuisance parameter. If "norm", it will affect the overall normalization of one or multiple
            samples in the process. If "scale", the nuisance parameter effect will be determined by varying
            factorization or regularization scales (depending on scale_variation and scales). If "pdf", the effect
            of the nuisance parameters will be determined by varying the PDF used.

        systematic_name : None or str, optional

        scale : {"mu", "mur", "muf"}, optional
            If type is "scale", this sets whether only the regularization scale ("mur"), only the factorization scale
            ("muf"), or both simultaneously ("mu") are varied. Default value: "mu".

        norm_variation : float, optional
            If type is "norm", this sets the relative effect of the nuisance parameter on the cross section at the
            "plus 1 sigma" variation. 1.1 corresponds to a 10% increase, 0.9 to a 10% decrease relative to the nominal
            cross section. Default value: 1.1.

        scale_variations : tuple of float, optional
            If type is "scale", this sets how the regularization and / or factorization scales are varied. A tuple
            like (0.5, 1.0, 2.0) specifies the factors with which they are varied. Default value: (0.5, 1.0, 2.0).

        pdf_variation : str, optional
            If type is "pdf", defines the PDF set for the variation. The option is passed along to the `--pdf` option
            of MadGraph's systematics module. See https://cp3.irmp.ucl.ac.be/projects/madgraph/wiki/Systematics for a
            list. The option "CT10" would, as an example, run over all the eigenvectors of the CTEQ10 set.
            Default value: "CT10".

        Returns
        -------
            None
        """

        assert scale in ["mu", "mur", "muf"]

        # Default name
        if systematic_name is None:
            i = 0
            while f"{effect}_{i}" in list(self.systematics.keys()):
                i += 1
            systematic_name = f"{type}_{i}"

        systematic_name = systematic_name.replace(" ", "_")
        systematic_name = systematic_name.replace("-", "_")

        scale = SystematicScale.from_str(scale)
        effect = SystematicType.from_str(effect)

        if effect is SystematicType.PDF:
            self.systematics[systematic_name] = Systematic(
                systematic_name,
                SystematicType.PDF,
                pdf_variation,
            )
        elif effect is SystematicType.SCALE:
            scale_variation_string = ",".join((str(factor) for factor in scale_variations))
            self.systematics[systematic_name] = Systematic(
                systematic_name,
                SystematicType.SCALE,
                scale_variation_string,
                scale,
            )
        elif effect is SystematicType.NORM:
            self.systematics[systematic_name] = Systematic(
                systematic_name,
                SystematicType.NORM,
                norm_variation,
            )

    def load(self, filename, disable_morphing=False):
        """
        Loads MadMiner setup from a file. All parameters, benchmarks, and morphing settings are overwritten.
        See `save` for more details.

        Parameters
        ----------
        filename : str
            Path to the MadMiner file.

        disable_morphing : bool, optional
            If True, the morphing setup is not loaded from the file. Default value: False.

        Returns
        -------
            None
        """

        # Load data
        (
            self.parameters,
            self.benchmarks,
            _,
            morphing_components,
            morphing_matrix,
            _,
            _,
            self.systematics,
            _,
            _,
            _,
            _,
            self.finite_difference_benchmarks,
            self.finite_difference_epsilon,
        ) = load_madminer_settings(filename, include_nuisance_benchmarks=False)

        logger.info("Found %s parameters:", len(self.parameters))
        for param in self.parameters.values():
            logger.info("   %s", param)

        logger.info("Found %s benchmarks:", len(self.benchmarks))
        for benchmark in self.benchmarks.values():
            logger.info("   %s", benchmark)

            if self.default_benchmark is None:
                self.default_benchmark = benchmark.name

        # Morphing
        self.morpher = None
        self.export_morphing = False

        if morphing_matrix is not None and morphing_components is not None and not disable_morphing:
            self.morpher = PhysicsMorpher(self.parameters)
            self.morpher.set_components(morphing_components)
            self.morpher.set_basis(self.benchmarks, morphing_matrix=morphing_matrix)
            self.export_morphing = True

            logger.info("Found morphing setup with %s components", len(morphing_components))
        else:
            logger.info("Did not find morphing setup.")

        # Systematics setup
        if len(self.systematics) == 0:
            logger.info("Did not find systematics setup.")
        else:
            logger.info("Found systematics setup with %s groups", len(self.systematics))
            for name, systematic in self.systematics.items():
                logger.debug("  %s: %s", name, systematic)

    def save(self, filename):
        """
        Saves MadMiner setup into a file.

        The file format follows the HDF5 standard. The saved information includes:

        * the parameter definitions,
        * the benchmark points,
        * the systematics setup (if defined), and
        * the morphing setup (if defined).

        This file is an important input to later stages in the analysis chain, including the processing of generated
        events, extraction of training samples, and calculation of Fisher information matrices. In these downstream
        tasks, additional information will be written to the MadMiner file, including the observations and event
        weights.

        Parameters
        ----------
        filename : str
            Path to the MadMiner file.

        Returns
        -------
            None

        """

        Path(filename).parent.mkdir(parents=True, exist_ok=True)

        if self.morpher is not None:
            logger.info("Saving setup (including morphing) to %s", filename)

            save_madminer_settings(
                file_name=filename,
                file_override=True,
                parameters=self.parameters,
                benchmarks=self.benchmarks,
                morphing_components=self.morpher.components,
                morphing_matrix=self.morpher.morphing_matrix,
                systematics=self.systematics,
                finite_differences=self.finite_difference_benchmarks,
                finite_differences_epsilon=self.finite_difference_epsilon,
            )
        else:
            logger.info("Saving setup (without morphing) to %s", filename)

            save_madminer_settings(
                file_name=filename,
                file_override=True,
                parameters=self.parameters,
                benchmarks=self.benchmarks,
                systematics=self.systematics,
                finite_differences=self.finite_difference_benchmarks,
                finite_differences_epsilon=self.finite_difference_epsilon,
            )

    def _export_cards(
        self,
        param_card_template_file,
        mg_process_directory,
        sample_benchmark=None,
        param_card_filename=None,
        reweight_card_filename=None,
        include_param_card=True,
        benchmarks=None,
    ):

        """
        Writes out a param_card and reweight_card for MadGraph. Instead of this low-level function, it is recommended to
        use `run` or `run_multiple`.

        Parameters
        ----------
        param_card_template_file : str
            Path to a param_card.dat of the used model.

        mg_process_directory : str
            Path to the directory of the MG process.

        sample_benchmark : str or None, optional
            Name of the benchmark used for sampling. If None, the very first defined benchmark is used. Default value:
            None.

        param_card_filename : str or None, optional
            Output filename for the generated param card. If None, a default filename in the MG process folder is used.
            Default value: None.

        reweight_card_filename : str or None, optional
            str or None. Output filename for the generated reweight card. If None, a default filename in the MG process
            folder is used. Default value: None.

        include_param_card : bool, optional
            If False, no param card is exported, only a reweight card

        benchmarks : None or OrderedDict, optional
            If None, uses all benchmarks. Otherwise uses these benchmarks.

        Returns
        -------
            None

        """

        if param_card_filename is None or reweight_card_filename is None:
            logger.info("Creating param and reweight cards in %s", mg_process_directory)
        else:
            logger.info("Creating param and reweight cards in %s, %s", param_card_filename, reweight_card_filename)

        if benchmarks is None:
            benchmarks = self.benchmarks

        # Check status
        assert self.default_benchmark is not None
        assert len(self.benchmarks) > 0

        # Default benchmark
        if sample_benchmark is None:
            sample_benchmark = self.default_benchmark

        # Export param card
        if include_param_card:
            export_param_card(
                benchmark=benchmarks[sample_benchmark],
                parameters=self.parameters,
                param_card_template_file=param_card_template_file,
                mg_process_directory=mg_process_directory,
                param_card_filename=param_card_filename,
            )

        # Export reweight card
        export_reweight_card(
            sample_benchmark=sample_benchmark,
            benchmarks=benchmarks,
            parameters=self.parameters,
            mg_process_directory=mg_process_directory,
            reweight_card_filename=reweight_card_filename,
        )

    def run(
        self,
        mg_directory,
        proc_card_file,
        param_card_template_file,
        run_card_file=None,
        mg_process_directory=None,
        pythia8_card_file=None,
        configuration_file=None,
        sample_benchmark=None,
        is_background=False,
        only_prepare_script=False,
        ufo_model_directory=None,
        log_directory=None,
        temp_directory=None,
        initial_command=None,
        systematics=None,
        order="LO",
        python_executable=None,
    ):

        """
        High-level function that creates the the MadGraph process, all required cards, and prepares or runs the event
        generation for one combination of cards.

        If `only_prepare_scripts=True`, the event generation is not run
        directly, but a bash script is created in `<process_folder>/madminer/run.sh` that will start the event
        generation with the correct settings.

        High-level function that creates the the MadGraph process, all required cards, and prepares or runs the event
        generation for multiple combinations of run_cards or importance samplings (`sample_benchmarks`).

        If `only_prepare_scripts=True`, the event generation is not run
        directly, but a bash script is created in `<process_folder>/madminer/run.sh` that will start the event
        generation with the correct settings.

        Parameters
        ----------
        mg_directory : str
            Path to the MadGraph 5 base directory.

        proc_card_file : str
            Path to the process card that tells MadGraph how to generate the process.

        param_card_template_file : str
            Path to a param card that will be used as template to create the
            appropriate param cards for these runs.

        run_card_file : str
            Paths to the MadGraph run card. If None, the default run_card is used.

        mg_process_directory : str or None, optional
            Path to the MG process directory. If None, MadMiner uses ./MG_process. Default value: None.

        pythia8_card_file : str or None, optional
            Path to the MadGraph Pythia8 card. If None, the card present in the process folder is used.
            Default value: None.

        configuration_file : str, optional
            Path to the MadGraph me5_configuration card. If None, the card present in the process folder
            is used. Default value: None.

        sample_benchmark : list of str or None, optional
            Lists the names of benchmarks that should be used to sample events. A different sampling does not change
            the expected differential cross sections, but will change which regions of phase space have many events
            (small variance) or few events (high variance). If None, the benchmark added first is used. Default value:
            None.

        is_background : bool, optional
            Should be True for background processes, i.e. process in which the differential cross section does not
            depend on the parameters (i.e. is the same for all benchmarks). In this case, no reweighting is run, which
            can substantially speed up the event generation. Default value: False.

        only_prepare_script : bool, optional
            If True, the event generation is not started, but instead a run.sh script is created in the process
            directory. Default value: False.

        ufo_model_directory : str or None, optional
            Path to an UFO model directory that should be used, but is not yet installed in mg_directory/models. The
            model will be copied to the MadGraph model directory before the process directory is generated. (Default
            value = None.

        log_directory : str or None, optional
            Directory for log files with the MadGraph output. If None, ./logs is used. Default value: None.

        temp_directory : str or None, optional
            Path to a temporary directory. If None, a system default is used. Default value: None.

        initial_command : str or None, optional
            Initial shell commands that have to be executed before MG is run (e.g. to load a virtual environment).
            Default value: None.

        systematics : None or list of str, optional
            If list of str, defines which systematics are used for this run.

        order : 'LO' or 'NLO', optional
            Differentiates between LO and NLO order runs. Minor changes to writing, reading and naming cards.
            Default value: 'LO'

        python_executable : None or str, optional
            Provides a path to the Python executable that should be used to call MadMiner. Default: None.

        Returns
        -------
            None

        """

        if sample_benchmark is None:
            sample_benchmark = self.default_benchmark

        self.run_multiple(
            mg_directory=mg_directory,
            proc_card_file=proc_card_file,
            param_card_template_file=param_card_template_file,
            run_card_files=[run_card_file],
            mg_process_directory=mg_process_directory,
            pythia8_card_file=pythia8_card_file,
            configuration_file=configuration_file,
            sample_benchmarks=[sample_benchmark],
            is_background=is_background,
            only_prepare_script=only_prepare_script,
            ufo_model_directory=ufo_model_directory,
            log_directory=log_directory,
            temp_directory=temp_directory,
            initial_command=initial_command,
            systematics=systematics,
            order=order,
            python_executable=python_executable,
        )

    def run_multiple(
        self,
        mg_directory,
        proc_card_file,
        param_card_template_file,
        run_card_files,
        mg_process_directory=None,
        pythia8_card_file=None,
        configuration_file=None,
        sample_benchmarks=None,
        is_background=False,
        only_prepare_script=False,
        ufo_model_directory=None,
        log_directory=None,
        temp_directory=None,
        initial_command=None,
        systematics=None,
        order="LO",
        python_executable=None,
    ):

        """
        High-level function that creates the the MadGraph process, all required cards, and prepares or runs the event
        generation for multiple combinations of run_cards or importance samplings (`sample_benchmarks`).

        If `only_prepare_scripts=True`, the event generation is not run
        directly, but a bash script is created in `<process_folder>/madminer/run.sh` that will start the event
        generation with the correct settings.

        Parameters
        ----------
        mg_directory : str
            Path to the MadGraph 5 base directory.

        proc_card_file : str
            Path to the process card that tells MadGraph how to generate the process.

        param_card_template_file : str
            Path to a param card that will be used as template to create the appropriate param cards for these runs.

        run_card_files : list of str
            Paths to the MadGraph run card.

        mg_process_directory : str or None, optional
            Path to the MG process directory. If None, MadMiner uses ./MG_process. Default value: None.

        pythia8_card_file : str, optional
            Path to the MadGraph Pythia8 card. If None, the card present in the process folder
            is used. Default value: None.

        configuration_file : str, optional
            Path to the MadGraph me5_configuration card. If None, the card present in the process folder
            is used. Default value: None.

        sample_benchmarks : list of str or None, optional
            Lists the names of benchmarks that should be used to sample events. A different sampling does not change
            the expected differential cross sections, but will change which regions of phase space have many events
            (small variance) or few events (high variance). If None, a run is started for each of the benchmarks, which
            should map out all regions of phase space well. Default value: None.

        is_background : bool, optional
            Should be True for background processes, i.e. process in which the differential cross section does not
            depend on the parameters (i.e. is the same for all benchmarks). In this case, no reweighting is run, which
            can substantially speed up the event generation. Default value: False.

        only_prepare_script : bool, optional
            If True, the event generation is not started, but instead a run.sh script is created in the process
            directory. Default value: False.

        ufo_model_directory : str or None, optional
            Path to an UFO model directory that should be used, but is not yet installed in mg_directory/models. The
            model will be copied to the MadGraph model directory before the process directory is generated. (Default
            value = None)

        log_directory : str or None, optional
            Directory for log files with the MadGraph output. If None, ./logs is used. Default value: None.

        temp_directory : str or None, optional
            Path to a temporary directory. If None, a system default is used. Default value: None.

        initial_command : str or None, optional
            Initial shell commands that have to be executed before MG is run (e.g. to load a virtual environment).
            If not specified and `python2_override` is True, it adds the user-installed Python2 binaries to the PATH.
            Default value: None.

        systematics : None or list of str, optional
            If list of str, defines which systematics are used for these runs.

        order : 'LO' or 'NLO', optional
            Differentiates between LO and NLO order runs. Minor changes to writing, reading and naming cards.
            Default value: 'LO'

        python_executable : None or str, optional
            Provides a path to the Python executable that should be used to call MadMiner. Default: None.

        Returns
        -------
            None

        """

        # Defaults
        if mg_process_directory is None:
            mg_process_directory = "./MG_process"

        if temp_directory is None:
            temp_directory = tempfile.gettempdir()

        if log_directory is None:
            log_directory = "./logs"

        if sample_benchmarks is None:
            sample_benchmarks = [benchmark for benchmark in self.benchmarks.keys()]

        # This snippet is useful when using virtual envs.
        # (Derives from a Python2 - Python3 issue).
        # Ref: https://github.com/madminer-tool/madminer/issues/422
        if python_executable and initial_command is None:
            logger.info(f"Adding {python_executable} bin folder to PATH")
            binary_path = os.popen(f"command -v {python_executable}").read().strip()
            binary_folder = Path(binary_path).parent

            initial_command = f"export PATH={binary_folder}:$PATH"
            logger.info(f"Using Python executable {binary_path}")

        # Generate process folder
        log_file_generate = f"{log_directory}/generate.log"

        generate_mg_process(
            mg_directory,
            temp_directory,
            proc_card_file,
            mg_process_directory,
            ufo_model_directory=ufo_model_directory,
            initial_command=initial_command,
            log_file=log_file_generate,
            python_executable=python_executable,
        )

        # Make MadMiner folders
        Path(mg_process_directory, "madminer", "cards").mkdir(parents=True, exist_ok=True)
        Path(mg_process_directory, "madminer", "scripts").mkdir(parents=True, exist_ok=True)

        # Systematics
        if systematics is None:
            systematics_used = self.systematics
        else:
            systematics_used = OrderedDict()
            for key in systematics:
                systematics_used[key] = self.systematics[key]

        # Loop over settings
        i = 0
        mg_scripts = []

        for run_card_file in run_card_files:
            for sample_benchmark in sample_benchmarks:

                # Files
                script_file = f"madminer/scripts/run_{i}.sh"
                log_file_run = f"run_{i}.log"
                mg_commands_filename = f"madminer/cards/mg_commands_{i}.dat"
                param_card_file = f"madminer/cards/param_card_{i}.dat"
                reweight_card_file = f"madminer/cards/reweight_card_{i}.dat"
                new_pythia8_card_file = None
                if pythia8_card_file is not None:
                    new_pythia8_card_file = f"madminer/cards/pythia8_card_{i}.dat"
                new_run_card_file = None
                if run_card_file is not None:
                    new_run_card_file = f"madminer/cards/run_card_{i}.dat"
                new_configuration_file = None
                if configuration_file is not None:
                    new_configuration_file = f"madminer/cards/me5_configuration_{i}.txt"

                logger.info("Run %s", i)
                logger.info("  Sampling from benchmark: %s", sample_benchmark)
                logger.info("  Original run card:       %s", run_card_file)
                logger.info("  Original Pythia8 card:   %s", pythia8_card_file)
                logger.info("  Original config card:    %s", configuration_file)
                logger.info("  Copied run card:         %s", new_run_card_file)
                logger.info("  Copied Pythia8 card:     %s", new_pythia8_card_file)
                logger.info("  Copied config card:      %s", new_configuration_file)
                logger.info("  Param card:              %s", param_card_file)
                logger.info("  Reweight card:           %s", reweight_card_file)
                logger.info("  Log file:                %s", log_file_run)

                # Check input
                if run_card_file is None and any(
                    syst.type in {SystematicType.PDF, SystematicType.SCALE} for syst in systematics_used.values()
                ):
                    logger.warning(
                        "Warning: No run card given, but PDF or scale variation set up. The correct systematics"
                        " settings are not set automatically. Make sure to set them correctly!"
                    )

                # Create param and reweight cards
                self._export_cards(
                    param_card_template_file,
                    mg_process_directory,
                    sample_benchmark=sample_benchmark,
                    param_card_filename=f"{mg_process_directory}/{param_card_file}",
                    reweight_card_filename=f"{mg_process_directory}/{reweight_card_file}",
                )

                # Create run card
                if run_card_file is not None:
                    export_run_card(
                        template_filename=run_card_file,
                        run_card_filename=f"{mg_process_directory}/{new_run_card_file}",
                        systematics=systematics_used,
                        order=order,
                    )

                # Copy Pythia card
                if pythia8_card_file is not None:
                    copy_file(pythia8_card_file, f"{mg_process_directory}/{new_pythia8_card_file}")

                # Copy Configuration card
                if configuration_file is not None:
                    copy_file(configuration_file, f"{mg_process_directory}/{new_configuration_file}")

                # Run MG and Pythia
                if only_prepare_script:
                    mg_script = setup_mg_with_scripts(
                        mg_process_directory,
                        proc_card_filename_from_mgprocdir=mg_commands_filename,
                        run_card_file_from_mgprocdir=new_run_card_file,
                        param_card_file_from_mgprocdir=param_card_file,
                        reweight_card_file_from_mgprocdir=reweight_card_file,
                        pythia8_card_file_from_mgprocdir=new_pythia8_card_file,
                        configuration_file_from_mgprocdir=new_configuration_file,
                        is_background=is_background,
                        script_file_from_mgprocdir=script_file,
                        initial_command=initial_command,
                        log_dir=log_directory,
                        log_file_from_logdir=log_file_run,
                        python_executable=python_executable,
                        order=order,
                    )
                    mg_scripts.append(mg_script)
                else:
                    run_mg(
                        mg_directory,
                        mg_process_directory,
                        f"{mg_process_directory}/{mg_commands_filename}",
                        f"{mg_process_directory}/{new_run_card_file}",
                        f"{mg_process_directory}/{param_card_file}",
                        f"{mg_process_directory}/{reweight_card_file}",
                        None if new_pythia8_card_file is None else f"{mg_process_directory}/{new_pythia8_card_file}",
                        None if new_configuration_file is None else f"{mg_process_directory}/{new_configuration_file}",
                        is_background=is_background,
                        initial_command=initial_command,
                        log_file=f"{log_directory}/{log_file_run}",
                        python_executable=python_executable,
                        order=order,
                    )

                i += 1

        n_runs_total = i

        # Master shell script
        if only_prepare_script:
            master_script_filename = f"{mg_process_directory}/madminer/run.sh"
            create_master_script(
                log_directory,
                master_script_filename,
                mg_directory,
                mg_process_directory,
                mg_scripts,
            )
            logger.info(
                "To generate events, please run:\n\n %s [MG_directory] [MG_process_directory] [log_dir]\n\n",
                master_script_filename,
            )

        else:
            expected_event_files = [f"{mg_process_directory}/Events/run_{(i+1):02d}" for i in range(n_runs_total)]
            expected_event_files = "\n".join(expected_event_files)
            logger.info(
                "Finished running MadGraph! Please check that events were successfully generated in the following "
                "folders:\n\n%s\n\n",
                expected_event_files,
            )

    def reweight_existing_sample(
        self,
        mg_process_directory,
        run_name,
        param_card_template_file,
        sample_benchmark,
        reweight_benchmarks=None,
        only_prepare_script=False,
        log_directory=None,
        initial_command=None,
    ):
        """
        High-level function that adds the weights required for MadMiner to an existing sample.

        If `only_prepare_scripts=True`, the event generation is not run
        directly, but a bash script is created in `<process_folder>/madminer/run.sh` that will start the event
        generation with the correct settings.

        Currently does not support adding systematics.

        Parameters
        ----------
        mg_process_directory : str
            Path to the MG process directory. If None, MadMiner uses ./MG_process.

        run_name : str
            Run name.

        param_card_template_file : str
            Path to a param card that will be used as template to create the appropriate param cards for these runs.

        sample_benchmark : str
            The name of the benchmark used to generate this sample.

        reweight_benchmarks : list of str or None
            Lists the names of benchmarks to which the sample should be reweighted. If None, all benchmarks (except
            sample_benchmarks) are used.

        only_prepare_script : bool, optional
            If True, the event generation is not started, but instead a run.sh script is created in the process
            directory. Default value: False.

        log_directory : str or None, optional
            Directory for log files with the MadGraph output. If None, ./logs is used. Default value: None.

        initial_command : str or None, optional
            Initial shell commands that have to be executed before MG is run (e.g. to load a virtual environment).
            Default value: None.

        Returns
        -------
            None

        """

        # TODO: check that we don't reweight to benchmarks that already have weights in the LHE file
        # TODO: add systematics

        # Defaults
        if log_directory is None:
            log_directory = "./logs"

        # Make MadMiner folders
        Path(mg_process_directory, "madminer", "cards").mkdir(parents=True, exist_ok=True)
        Path(mg_process_directory, "madminer", "scripts").mkdir(parents=True, exist_ok=True)

        # Files
        script_file = "madminer/scripts/run_reweight.sh"
        log_file_run = "reweight.log"
        reweight_card_file = "/madminer/cards/reweight_card_reweight.dat"

        # Missing benchmarks
        missing_benchmarks = OrderedDict()
        for benchmark_name in reweight_benchmarks:
            missing_benchmarks[benchmark_name] = self.benchmarks[benchmark_name]

        # Inform user
        logger.info("Reweighting setup")
        logger.info("  Originally sampled from benchmark: %s", sample_benchmark)
        logger.info("  Now reweighting to benchmarks:     %s", reweight_benchmarks)
        logger.info("  Reweight card:                     %s", reweight_card_file)
        logger.info("  Log file:                          %s", log_file_run)

        # Create param and reweight cards
        self._export_cards(
            param_card_template_file,
            mg_process_directory,
            sample_benchmark=sample_benchmark,
            reweight_card_filename=f"{mg_process_directory}/{reweight_card_file}",
            include_param_card=False,
            benchmarks=missing_benchmarks,
        )

        # Run reweighting
        if only_prepare_script:
            call_instruction = setup_mg_reweighting_with_scripts(
                mg_process_directory,
                run_name=run_name,
                reweight_card_file_from_mgprocdir=reweight_card_file,
                script_file_from_mgprocdir=script_file,
                initial_command=initial_command,
                log_dir=log_directory,
                log_file_from_logdir=log_file_run,
            )

            logger.info("To generate events, please run:\n\n %s \n\n", call_instruction)

        else:
            run_mg_reweighting(
                mg_process_directory,
                run_name=run_name,
                reweight_card_file=f"{mg_process_directory}/{reweight_card_file}",
                initial_command=initial_command,
                log_file=f"{log_directory}/{log_file_run}",
            )
            logger.info(
                "Finished running reweighting! Please check that events were successfully reweighted in the following "
                "folder:\n\n %s/Events/%s \n\n",
                mg_process_directory,
                run_name,
            )
