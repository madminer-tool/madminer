from __future__ import absolute_import, division, print_function, unicode_literals
import six

import os
import logging
from collections import OrderedDict
import tempfile

from madminer.utils.morphing import PhysicsMorpher
from madminer.utils.interfaces.madminer_hdf5 import save_madminer_settings, load_madminer_settings
from madminer.utils.interfaces.mg_cards import export_param_card, export_reweight_card, export_run_card
from madminer.utils.interfaces.mg import generate_mg_process, setup_mg_with_scripts, run_mg, create_master_script
from madminer.utils.various import create_missing_folders, format_benchmark, copy_file

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
        self.systematics = None

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

        morphing_max_power : int or tuple of int
            The maximal power with which this parameter contributes to the
            squared matrix element of the process of interest. If a tuple is given, gives this
            maximal power for each of several operator configurations. Typically at tree level,
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
            parameter_name = "parameter_" + str(len(self.parameters))

        # Check and sanitize input
        assert isinstance(parameter_name, six.string_types), "Parameter name is not a string: {}".format(parameter_name)
        assert isinstance(lha_block, six.string_types), "LHA block is not a string: {}".format(lha_block)
        assert isinstance(lha_id, int), "LHA id is not an integer: {}".format(lha_id)

        parameter_name = parameter_name.replace(" ", "_")
        parameter_name = parameter_name.replace("-", "_")

        assert parameter_name not in self.parameters, "Parameter name exists already: {}".format(parameter_name)

        if isinstance(morphing_max_power, int):
            morphing_max_power = (morphing_max_power,)

        # Add parameter
        self.parameters[parameter_name] = (lha_block, lha_id, morphing_max_power, parameter_range, param_card_transform)

        # After manually adding parameters, the morphing information is not accurate anymore
        self.morpher = None

        logger.info(
            "Added parameter %s (LHA: %s %s, maximal power in squared ME: %s, range: %s)",
            parameter_name,
            lha_block,
            lha_id,
            morphing_max_power,
            parameter_range,
        )

        # Reset benchmarks
        if len(self.benchmarks) > 0:
            logger.warning("Resetting benchmarks and morphing")

        self.benchmarks = OrderedDict()
        self.default_benchmark = None
        self.morpher = None
        self.export_morphing = False

    def set_parameters(self, parameters=None):

        """
        Manually sets all parameters, overwriting previously added parameters.

        Parameters
        ----------
        parameters : dict or list or None, optional
             If parameters is None, resets parameters. If parameters is an dict, the keys should be str and give the
             parameter names, and the values are tuples of the form (LHA_block, LHA_ID, morphing_max_power, param_min,
             param_max) or of the form (LHA_block, LHA_ID). If parameters is a list, the items should be tuples of the
             form (LHA_block, LHA_ID). Default value: None.

        Returns
        -------
            None

        """

        if parameters is None:
            parameters = OrderedDict()

        self.parameters = OrderedDict()

        if isinstance(parameters, dict):
            for key, values in six.iteritems(parameters):
                if len(values) == 5:
                    self.add_parameter(
                        lha_block=values[0],
                        lha_id=values[1],
                        parameter_name=key,
                        parameter_range=[values[3], values[4]],
                        morphing_max_power=values[2],
                    )
                elif len(values) == 2:
                    self.add_parameter(lha_block=values[0], lha_id=values[1], parameter_name=key)
                else:
                    raise ValueError("Parameter properties has unexpected length: {0}".format(values))

        else:
            for values in parameters:
                assert len(values) == 2, "Parameter list entry does not have length 2: {0}".format(values)
                self.add_parameter(values[0], values[1])

        # After manually adding parameters, the morphing information is not accurate anymore
        if len(self.benchmarks) > 0:
            logger.warning("Resetting benchmarks and morphing")

        self.benchmarks = OrderedDict()
        self.default_benchmark = None
        self.morpher = None
        self.export_morphing = False

    def add_benchmark(self, parameter_values, benchmark_name=None, verbose=True):
        """
        Manually adds an individual benchmark, that is, a parameter point that will be evaluated by MadGraph.

        If this command is called before

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
            benchmark_name = "benchmark_" + str(len(self.benchmarks))

        # Check input
        if not isinstance(parameter_values, dict):
            raise RuntimeError("Parameter values are not a dict: {}".format(parameter_values))

        for key, value in six.iteritems(parameter_values):
            if key not in self.parameters:
                raise RuntimeError("Unknown parameter: {0}".format(key))

        if benchmark_name in self.benchmarks:
            raise RuntimeError("Benchmark name {} exists already".format(benchmark_name))

        # Add benchmark
        self.benchmarks[benchmark_name] = parameter_values

        # If first benchmark, this will be the default for sampling
        if len(self.benchmarks) == 1:
            self.default_benchmark = benchmark_name

        if verbose:
            logger.info("Added benchmark %s: %s)", benchmark_name, format_benchmark(parameter_values))
        else:
            logger.debug("Added benchmark %s: %s)", benchmark_name, format_benchmark(parameter_values))

    def set_benchmarks(self, benchmarks=None, verbose=True):
        """
        Manually sets all benchmarks, that is, parameter points that will be evaluated by MadGraph. Calling this
        function overwrites all previously defined benchmarks.

        Parameters
        ----------
        benchmarks : dict or list or None, optional
            Specifies all benchmarks. If None, all benchmarks are reset. If dict, the keys are the benchmark names and
            the values are dicts of the form {parameter_name:value}. If list, the entries are dicts
            {parameter_name:value} (and the benchmark names are chosen automatically). Default value: None.

        verbose : bool, optional
            If True, prints output about each benchmark. Default value: True.

        Returns
        -------
            None

        """

        if benchmarks is None:
            benchmarks = OrderedDict()

        self.benchmarks = OrderedDict()
        self.default_benchmark = None

        if isinstance(benchmarks, dict):
            for name, values in six.iteritems(benchmarks):
                self.add_benchmark(values, name, verbose=verbose)
        else:
            for values in benchmarks:
                self.add_benchmark(values)

        # After manually adding benchmarks, the morphing information is not accurate anymore
        if self.morpher is not None:
            logger.warning("Reset morphing")
            self.morpher = None
            self.export_morphing = False

    def set_morphing(
        self, max_overall_power=4, n_bases=1, include_existing_benchmarks=True, n_trials=100, n_test_thetas=100
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
        max_overall_power : int or tuple of int, optional
            The maximal sum of powers of all parameters contributing to the squared matrix element. If a tuple is given,
            gives the maximal sum of powers for each of several operator configurations (see `add_parameter`).
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

        if isinstance(max_overall_power, int):
            max_overall_power = (max_overall_power,)

        morpher = PhysicsMorpher(parameters_from_madminer=self.parameters)
        morpher.find_components(max_overall_power)

        if include_existing_benchmarks:
            n_predefined_benchmarks = len(self.benchmarks)
            basis = morpher.optimize_basis(
                n_bases=n_bases,
                fixed_benchmarks_from_madminer=self.benchmarks,
                n_trials=n_trials,
                n_test_thetas=n_test_thetas,
            )
        else:
            n_predefined_benchmarks = 0
            basis = morpher.optimize_basis(
                n_bases=n_bases, fixed_benchmarks_from_madminer=None, n_trials=n_trials, n_test_thetas=n_test_thetas
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

    def set_systematics(self, scale_variation=None, scales="together", pdf_variation=None):
        """
        Prepares the simulation of the effect of different nuisance parameters, including scale variations and PDF
        changes.

        Parameters
        ----------
        scale_variation : None or tuple of float, optional
            If not None, the regularization and / or factorization scales are varied. A tuple like (0.5,1.,2.)
            specifies the factors with which they are varied. Default value: None.

        scales : {"together", "independent", "mur", "muf"}, optional
            Whether only the regularization scale ("mur"), only the factorization scale ("muf"), both simultanously
            ("together") or both independently ("independent") are varied. Default value: "together".

        pdf_variation : None or str, optional
            If not None, the PDFs are varied. The option is passed along to the `--pdf` option
            of MadGraph's systematics module. See https://cp3.irmp.ucl.ac.be/projects/madgraph/wiki/Systematics for a
            list. The option "CT10" would, as an example, run over all the eigenvectors of the CTEQ10 set.

        Returns
        -------
            None

        """

        # Check input
        if scales not in ["together", "independent", "mur", "muf"]:
            raise ValueError("Unknown value {} for argument scales".format(scales))

        # Save systematics setup
        self.systematics = OrderedDict()

        if scale_variation is not None:
            scale_variation_string = ",".join([str(factor) for factor in scale_variation])

            if scales == "together":
                self.systematics["mu"] = scale_variation_string
            elif scales == "independent":
                self.systematics["muf"] = scale_variation_string
                self.systematics["mur"] = scale_variation_string
            elif scales == "mur":
                self.systematics["mur"] = scale_variation_string
            elif scales == "muf":
                self.systematics["muf"] = scale_variation_string

        if pdf_variation is not None:
            self.systematics["pdf"] = pdf_variation

        if len(self.systematics) == 0:
            self.systematics = None

    def load(self, filename, disable_morphing=False):
        """
        Loads MadMiner setup from a file. All parameters, benchmarks, and morphing settings are overwritten. See `save`
        for more details.

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
        ) = load_madminer_settings(filename, include_nuisance_benchmarks=False)

        logger.info("Found %s parameters:", len(self.parameters))
        for key, values in six.iteritems(self.parameters):
            logger.info(
                "   %s (LHA: %s %s, maximal power in squared ME: %s, range: %s)",
                key,
                values[0],
                values[1],
                values[2],
                values[3],
            )

        logger.info("Found %s benchmarks:", len(self.benchmarks))
        for key, values in six.iteritems(self.benchmarks):
            logger.info("   %s: %s", key, format_benchmark(values))

            if self.default_benchmark is None:
                self.default_benchmark = key

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
        if self.systematics is None:
            logger.info("Did not find systematics setup.")
        else:
            logger.info("Found systematics setup with %s nuisance parameter groups", len(self.systematics))

            for key, value in six.iteritems(self.systematics):
                logger.debug("  %s: %s", key, value)

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

        create_missing_folders([os.path.dirname(filename)])

        if self.morpher is not None:
            logger.info("Saving setup (including morphing) to %s", filename)

            save_madminer_settings(
                filename=filename,
                parameters=self.parameters,
                benchmarks=self.benchmarks,
                morphing_components=self.morpher.components,
                morphing_matrix=self.morpher.morphing_matrix,
                systematics=self.systematics,
                overwrite_existing_files=True,
            )
        else:
            logger.info("Saving setup (without morphing) to %s", filename)

            save_madminer_settings(
                filename=filename,
                parameters=self.parameters,
                benchmarks=self.benchmarks,
                systematics=self.systematics,
                overwrite_existing_files=True,
            )

    def _export_cards(
        self,
        param_card_template_file,
        mg_process_directory,
        sample_benchmark=None,
        param_card_filename=None,
        reweight_card_filename=None,
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

        Returns
        -------
            None

        """

        if param_card_filename is None or reweight_card_filename is None:
            logger.info("Creating param and reweight cards in %s", mg_process_directory)
        else:
            logger.info("Creating param and reweight cards in %s, %s", param_card_filename, reweight_card_filename)

        # Check status
        assert self.default_benchmark is not None
        assert len(self.benchmarks) > 0

        # Default benchmark
        if sample_benchmark is None:
            sample_benchmark = self.default_benchmark

        # Export param card
        export_param_card(
            benchmark=self.benchmarks[sample_benchmark],
            parameters=self.parameters,
            param_card_template_file=param_card_template_file,
            mg_process_directory=mg_process_directory,
            param_card_filename=param_card_filename,
        )

        # Export reweight card
        export_reweight_card(
            sample_benchmark=sample_benchmark,
            benchmarks=self.benchmarks,
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
        sample_benchmark=None,
        is_background=False,
        only_prepare_script=False,
        ufo_model_directory=None,
        log_directory=None,
        temp_directory=None,
        initial_command=None,
        python2_override=False,
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

        only_prepare_script : bool, optional
            If True, MadGraph is not executed, but instead a run.sh script is created in
            the process directory. Default value: False.

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

        python2_override : bool, optional
            If True, MadMiner explicitly calls "python2" instead of relying on the system Python version to be
            Python 2.6 or Python 2.7. If you use systematics, make sure that the python interface of LHAPDF was compiled
            with the Python version you are using. Default: False.

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
            sample_benchmarks=[sample_benchmark],
            is_background=is_background,
            only_prepare_script=only_prepare_script,
            ufo_model_directory=ufo_model_directory,
            log_directory=log_directory,
            temp_directory=temp_directory,
            initial_command=initial_command,
            python2_override=python2_override,
        )

    def run_multiple(
        self,
        mg_directory,
        proc_card_file,
        param_card_template_file,
        run_card_files,
        mg_process_directory=None,
        pythia8_card_file=None,
        sample_benchmarks=None,
        is_background=False,
        only_prepare_script=False,
        ufo_model_directory=None,
        log_directory=None,
        temp_directory=None,
        initial_command=None,
        python2_override=False,
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

        only_prepare_script : bool, optional
            If True, MadGraph is not executed, but instead a run.sh script is created in
            the process directory. Default value: False.

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
            Default value: None.

        python2_override : bool, optional
            If True, MadMiner explicitly calls "python2" instead of relying on the system Python version to be
            Python 2.6 or Python 2.7. If you use systematics, make sure that the python interface of LHAPDF was compiled
            with the Python version you are using. Default: False.

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
            sample_benchmarks = [benchmark for benchmark in self.benchmarks]

        # Generate process folder
        log_file_generate = log_directory + "/generate.log"

        generate_mg_process(
            mg_directory,
            temp_directory,
            proc_card_file,
            mg_process_directory,
            ufo_model_directory=ufo_model_directory,
            initial_command=initial_command,
            log_file=log_file_generate,
            explicit_python_call=python2_override,
        )

        # Make MadMiner folders
        create_missing_folders(
            [
                mg_process_directory + "/madminer",
                mg_process_directory + "/madminer/cards",
                mg_process_directory + "/madminer/scripts",
            ]
        )

        # Loop over settings
        i = 0
        mg_scripts = []

        for run_card_file in run_card_files:
            for sample_benchmark in sample_benchmarks:

                # Files
                script_file = "madminer/scripts/run_{}.sh".format(i)
                log_file_run = "run_{}.log".format(i)
                mg_commands_filename = "/madminer/cards/mg_commands_{}.dat".format(i)
                param_card_file = "/madminer/cards/param_card_{}.dat".format(i)
                reweight_card_file = "/madminer/cards/reweight_card_{}.dat".format(i)
                new_pythia8_card_file = None
                if pythia8_card_file is not None:
                    new_pythia8_card_file = "/madminer/cards/pythia8_card_{}.dat".format(i)
                new_run_card_file = None
                if run_card_file is not None:
                    new_run_card_file = "/madminer/cards/run_card_{}.dat".format(i)

                logger.info("Run %s", i)
                logger.info("  Sampling from benchmark: %s", sample_benchmark)
                logger.info("  Original run card:       %s", run_card_file)
                logger.info("  Original Pythia8 card:   %s", pythia8_card_file)
                logger.info("  Copied run card:         %s", new_run_card_file)
                logger.info("  Copied Pythia8 card:     %s", new_pythia8_card_file)
                logger.info("  Param card:              %s", param_card_file)
                logger.info("  Reweight card:           %s", reweight_card_file)
                logger.info("  Log file:                %s", log_file_run)

                # Check input
                if run_card_file is None and self.systematics is not None:
                    logger.warning(
                        "Warning: No run card given, but systematics set up. The correct systematics"
                        " settings are not set automatically. Make sure to set them correctly!"
                    )

                # Create param and reweight cards
                self._export_cards(
                    param_card_template_file,
                    mg_process_directory,
                    sample_benchmark=sample_benchmark,
                    param_card_filename=mg_process_directory + "/" + param_card_file,
                    reweight_card_filename=mg_process_directory + "/" + reweight_card_file,
                )

                # Create run card
                if run_card_file is not None:
                    export_run_card(
                        template_filename=run_card_file,
                        run_card_filename=mg_process_directory + "/" + new_run_card_file,
                        systematics=self.systematics,
                    )

                # Copy Pythia card
                if pythia8_card_file is not None:
                    copy_file(pythia8_card_file, mg_process_directory + "/" + new_pythia8_card_file)

                # Run MG and Pythia
                if only_prepare_script:
                    mg_script = setup_mg_with_scripts(
                        mg_process_directory,
                        proc_card_filename_from_mgprocdir=mg_commands_filename,
                        run_card_file_from_mgprocdir=new_run_card_file,
                        param_card_file_from_mgprocdir=param_card_file,
                        reweight_card_file_from_mgprocdir=reweight_card_file,
                        pythia8_card_file_from_mgprocdir=None
                        if new_pythia8_card_file is None
                        else new_pythia8_card_file,
                        is_background=is_background,
                        script_file_from_mgprocdir=script_file,
                        initial_command=initial_command,
                        log_dir=log_directory,
                        log_file_from_logdir=log_file_run,
                        explicit_python_call=python2_override,
                    )
                    mg_scripts.append(mg_script)
                else:
                    run_mg(
                        mg_directory,
                        mg_process_directory,
                        mg_process_directory + "/" + mg_commands_filename,
                        mg_process_directory + "/" + new_run_card_file,
                        mg_process_directory + "/" + param_card_file,
                        mg_process_directory + "/" + reweight_card_file,
                        None if new_pythia8_card_file is None else mg_process_directory + "/" + new_pythia8_card_file,
                        is_background=is_background,
                        initial_command=initial_command,
                        log_file=log_directory + "/" + log_file_run,
                        explicit_python_call=python2_override,
                    )

                i += 1

        n_runs_total = i

        # Master shell script
        if only_prepare_script:
            master_script_filename = "{}/madminer/run.sh".format(mg_process_directory)
            create_master_script(log_directory, master_script_filename, mg_directory, mg_process_directory, mg_scripts)

            logger.info(
                "To generate events, please run:\n\n %s [MG_directory] [MG_process_directory] [log_dir]\n\n",
                master_script_filename,
            )

        else:
            expected_event_files = [
                mg_process_directory + "/Events/run_{:02d}".format(i + 1) for i in range(n_runs_total)
            ]
            expected_event_files = "\n".join(expected_event_files)
            logger.info(
                "Finished running MadGraph! Please check that events were succesfully generated in the following "
                "folders:\n\n%s\n\n",
                expected_event_files,
            )
