from __future__ import absolute_import, division, print_function, unicode_literals
import six

import os
import logging
from collections import OrderedDict
import tempfile

from madminer.morphing import AdvancedMorpher
from madminer.utils.interfaces.hdf5 import save_madminer_settings, load_madminer_settings
from madminer.utils.interfaces.mg_cards import export_param_card, export_reweight_card
from madminer.utils.interfaces.mg import generate_mg_process, run_mg_pythia, copy_ufo_model
from madminer.utils.various import create_missing_folders, general_init, format_benchmark


class MadMiner:

    def __init__(self, debug=False):

        """ Constructor """

        general_init(debug=debug)

        self.parameters = OrderedDict()
        self.benchmarks = OrderedDict()
        self.default_benchmark = None
        self.morpher = None
        self.export_morphing = False

    def add_parameter(self,
                      lha_block,
                      lha_id,
                      parameter_name=None,
                      param_card_transform=None,
                      morphing_max_power=2,
                      parameter_range=(0., 1.)):

        """ Adds an individual parameter

        :param lha_block: str, the name of the LHA block as used in the param_card. Case-sensitive.
        :param lha_id: int, the LHA id as used in the param_card.
        :param parameter_name: an internal name for the parameter. If None, a the default 'benchmark_i' is used.
        :param morphing_max_power: int or tuple of ints. The maximal power with which this parameter contributes to the
                                   squared matrix element of the process of interest. If a tuple is given, gives this
                                   maximal power for each of several operator configuraations. Typically at tree level,
                                   this maximal number is 2 for parameters that affect one vertex (e.g. only production
                                   or only decay of a particle), and 4 for parameters that affect two vertices (e.g.
                                   production and decay).
        :param param_card_transform: None or str that represents a one-parameter function mapping the parameter
                                     (`"theta"`) to the value that should be written in the parameter cards. This
                                     str is parsed by Python's `eval()` function, and `"theta"` is parsed as the
                                     parameter value.
        :param parameter_range: tuple, the range of parameter values of primary interest. Only affects the
                                         basis optimization.
        """

        # Default names
        if parameter_name is None:
            parameter_name = 'parameter_' + str(len(self.parameters))

        # Check and sanitize input
        assert isinstance(parameter_name, six.string_types), 'Parameter name is not a string: {}'.format(parameter_name)
        assert isinstance(lha_block, six.string_types), 'LHA block is not a string: {}'.format(lha_block)
        assert isinstance(lha_id, int), 'LHA id is not an integer: {}'.format(lha_id)

        parameter_name = parameter_name.replace(' ', '_')
        parameter_name = parameter_name.replace('-', '_')

        assert parameter_name not in self.parameters, 'Parameter name exists already: {}'.format(parameter_name)

        if isinstance(morphing_max_power, int):
            morphing_max_power = (morphing_max_power,)

        # Add parameter
        self.parameters[parameter_name] = (lha_block, lha_id,
                                           morphing_max_power, parameter_range, param_card_transform)

        # After manually adding parameters, the morphing information is not accurate anymore
        self.morpher = None

        logging.info('Added parameter %s (LHA: %s %s, maximal power in squared ME: %s, range: %s)', parameter_name,
                     lha_block, lha_id, morphing_max_power, parameter_range)

    def set_parameters(self,
                       parameters=None):

        """ Defines whole parameter space.

        :type parameters: dict or list that specifies all parameters. If dict, the keys are the parameter names and the
                          values describe the parameter properties. If list, the entries are tuples that describe the
                          parameter properties (the parameter names are chosen automatically). In both cases, the
                          parameter property tuples can have the form (LHA block, LHA id) or (LHA block, LHA id,
                          maximal power of this parameter in squared matrix element, minimum value, maximum value). The
                          last three parameters are only important for morphing.
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
                        morphing_max_power=values[2]
                    )
                elif len(values) == 2:
                    self.add_parameter(
                        lha_block=values[0],
                        lha_id=values[1],
                        parameter_name=key
                    )
                else:
                    raise ValueError('Parameter properties has unexpected length: {0}'.format(values))

        else:
            for values in parameters:
                assert len(values) == 2, 'Parameter list entry does not have length 2: {0}'.format(values)
                self.add_parameter(values[0], values[1])

        # After manually adding parameters, the morphing information is not accurate anymore
        self.morpher = None

    def add_benchmark(self,
                      parameter_values,
                      benchmark_name=None):
        """
        Adds an individual parameter benchmark.

        :param parameter_values: dict, with keys equal to parameter names and values equal to parameter values
        :param benchmark_name: str, defines name of benchmark
        """

        # Default names
        if benchmark_name is None:
            benchmark_name = 'benchmark_' + str(len(self.benchmarks))

        # Check input
        assert isinstance(parameter_values, dict), 'Parameter values are not a dict: {}'.format(parameter_values)

        for key, value in six.iteritems(parameter_values):
            assert key in self.parameters, 'Unknown parameter: {0}'.format(key)

        assert benchmark_name not in self.benchmarks, 'Benchmark name exists already: {}'.format(benchmark_name)

        # Add benchmark
        self.benchmarks[benchmark_name] = parameter_values

        # If first benchmark, this will be the default for sampling
        if len(self.benchmarks) == 1:
            self.default_benchmark = benchmark_name

        # After manually adding benchmarks, the morphing information is not accurate anymore
        self.morpher = None

        logging.info('Added benchmark %s: %s)', benchmark_name, format_benchmark(parameter_values))

    def set_benchmarks(self,
                       benchmarks=None):
        """
        Sets all parameter benchmarks.

        :param benchmarks: dict or list that specifies all parameters. If dict, the keys are the benchmark names and
                           the values are dicts {parameter_name:value}. If list, the entries are dicts
                           {parameter_name:value} (and the benchmark names are chosen automatically).
        """

        if benchmarks is None:
            benchmarks = OrderedDict()

        self.benchmarks = OrderedDict()
        self.default_benchmark = None

        if isinstance(benchmarks, dict):
            for name, values in six.iteritems(benchmarks):
                self.add_benchmark(values, name)
        else:
            for values in benchmarks:
                self.add_benchmark(values)

        # After manually adding benchmarks, the morphing information is not accurate anymore
        self.morpher = None

    def set_benchmarks_from_morphing(self,
                                     max_overall_power=4,
                                     n_bases=1,
                                     keep_existing_benchmarks=False,
                                     n_trials=100,
                                     n_test_thetas=100):
        """
        Sets all parameter benchmarks based on a morphing algorithm. The morphing basis is optimized with respect to the
        expected mean squared morphing weights over the parameter region of interest.

        :param max_overall_power: int or tuple of ints. The maximal sum of powers of all parameters, for each operator
                                  configuration (in the case of a tuple of ints).
        :param n_bases: The number of morphing bases generated. If n_bases > 1, multiple bases are combined, and the
                        weights for each basis are reduced by a factor 1 / n_bases. Currently not supported.
        :param keep_existing_benchmarks: Whether the previously defined benchmarks are included in the basis. In that
                                         case, the number of free parameters in the optimization routine is reduced.
        :param n_trials: Number of candidate bases used in the optimization.
        :param n_test_thetas: Number of validation benchmark points used to calculate the expected mean squared morphing
                              weight.
        """

        logging.info('Optimizing basis for morphing')

        if isinstance(max_overall_power, int):
            max_overall_power = (max_overall_power,)

        morpher = AdvancedMorpher(parameters_from_madminer=self.parameters)
        morpher.find_components(max_overall_power)

        if keep_existing_benchmarks:
            basis = morpher.optimize_basis(n_bases=n_bases,
                                           fixed_benchmarks_from_madminer=self.benchmarks,
                                           n_trials=n_trials,
                                           n_test_thetas=n_test_thetas)
        else:
            basis = morpher.optimize_basis(n_bases=n_bases,
                                           fixed_benchmarks_from_madminer=None,
                                           n_trials=n_trials,
                                           n_test_thetas=n_test_thetas)

        self.set_benchmarks(basis)
        self.morpher = morpher
        self.export_morphing = True

    def set_benchmarks_from_finite_differences(self):
        raise NotImplementedError

    def load(self, filename, disable_morphing=False):
        """ Loads MadMiner setup from HDF5 file """

        # Load data
        (self.parameters, self.benchmarks, morphing_components, morphing_matrix,
         _, _) = load_madminer_settings(filename)

        logging.info('Found %s parameters:', len(self.parameters))
        for key, values in six.iteritems(self.parameters):
            logging.info('   %s (LHA: %s %s, maximal power in squared ME: %s, range: %s)',
                         key, values[0], values[1], values[2], values[3])

        logging.info('Found %s benchmarks:', len(self.benchmarks))
        for key, values in six.iteritems(self.benchmarks):
            logging.info('   %s: %s',
                         key, format_benchmark(values))

            if self.default_benchmark is None:
                self.default_benchmark = key

        # Morphing
        self.morpher = None
        self.export_morphing = False

        if morphing_matrix is not None and morphing_components is not None and not disable_morphing:
            self.morpher = AdvancedMorpher(self.parameters)
            self.morpher.set_components(morphing_components)
            self.morpher.set_basis(self.benchmarks, morphing_matrix=morphing_matrix)
            self.export_morphing = True

            logging.info('Found morphing setup with %s components', len(morphing_components))

        else:
            logging.info('Did not find morphing setup.')

    def save(self, filename):
        """ Saves MadMiner setup from HDF5 file """

        create_missing_folders([os.path.dirname(filename)])

        if self.morpher is not None:
            logging.info('Saving setup (including morphing) to %s', filename)

            save_madminer_settings(filename=filename,
                                   parameters=self.parameters,
                                   benchmarks=self.benchmarks,
                                   morphing_components=self.morpher.components,
                                   morphing_matrix=self.morpher.morphing_matrix)
        else:
            logging.info('Saving setup (without morphing) to %s', filename)

            save_madminer_settings(filename=filename,
                                   parameters=self.parameters,
                                   benchmarks=self.benchmarks)

    @staticmethod
    def generate_mg_process(mg_directory,
                            temp_directory,
                            proc_card_file,
                            mg_process_directory,
                            ufo_model_directory=None,
                            initial_command=None,
                            log_file=None):

        """
        Lets MadGraph create the process folder.

        :param mg_directory: MadGraph 5 directory.
        :param temp_directory: A temporary directory.
        :param proc_card_file: Path to the process card that tells MadGraph how to generate the process.
        :param mg_process_directory: Path to the MG process directory.
        :param ufo_model_directory: Path to a non-standard UFO model, which will be copied to the MG directory before
                                    executing the process card.
        :param initial_command: Initial shell commands that have to be executed before MG is run (e.g. loading a virtual
                                environment).
        :param log_file: Path to a log file in which the MadGraph output is saved.
        """

        logging.info('Generating MadGraph process folder from %s at %s', proc_card_file, mg_process_directory)

        create_missing_folders([temp_directory, mg_process_directory, os.path.dirname(log_file)])

        if ufo_model_directory is not None:
            copy_ufo_model(ufo_model_directory, mg_directory)

        generate_mg_process(
            mg_directory,
            temp_directory,
            proc_card_file,
            mg_process_directory,
            initial_command=initial_command,
            log_file=log_file
        )

    def export_cards(self,
                     param_card_template_file,
                     reweight_card_template_file,
                     mg_process_directory,
                     sample_benchmark=None):

        """
        Writes out param_card and reweight_card for MadGraph. Currently, this is the final output of this package.
        Future versions are scheduled to support the automatic running of MadGraph, and the automated conversion of
        event files.

        :param param_card_template_file: Path to a param_card.dat of the used model.
        :param reweight_card_template_file: Path to an empty reweight_card.dat (no commands, the default 'launch' should
                                            be commented out or removed).
        :param mg_process_directory: Path to the directory of the MG process.
        :param sample_benchmark: Name of the benchmark used for sampling. If None, the very first defined benchmark is
                                 used.
        """

        logging.info('Creating param and reweight cards in %s', mg_process_directory)

        # Check status
        assert self.default_benchmark is not None
        assert len(self.benchmarks) > 0

        # Default benchmark
        if sample_benchmark is None:
            sample_benchmark = self.default_benchmark

        # Export param card
        export_param_card(benchmark=self.benchmarks[sample_benchmark],
                          parameters=self.parameters,
                          param_card_template_file=param_card_template_file,
                          mg_process_directory=mg_process_directory)

        # Export reweight card
        export_reweight_card(sample_benchmark=sample_benchmark,
                             benchmarks=self.benchmarks,
                             parameters=self.parameters,
                             reweight_card_template_file=reweight_card_template_file,
                             mg_process_directory=mg_process_directory)

    @staticmethod
    def run_mg_pythia(mg_directory,
                      mg_process_directory,
                      proc_card_filename=None,
                      run_card_file=None,
                      param_card_file=None,
                      reweight_card_file=None,
                      pythia8_card_file=None,
                      is_background=False,
                      only_prepare_script=False,
                      initial_command=None,
                      log_file=None):

        """
        Runs the event generation with MadGraph and Pythia.

        :param mg_directory: Path to the MadGraph 5 base directory.
        :param mg_process_directory: Path to the MG process directory.
        :param proc_card_filename: Filename for the process card that will be generated.
        :param run_card_file: Path to the MadGraph run card. If None, the card present in the process folder is used.
        :param param_card_file: Path to the MadGraph run card. If None, the card present in the process folder is used.
        :param reweight_card_file: Path to the MadGraph reweight card. If None, the card present in the process folder
                                   is used.
        :param pythia8_card_file: Path to the MadGraph Pythia8 card. If None, the card present in the process folder
                                  is used.
        :param is_background: bool that should be True for background processes, i.e. process in which the differential
                              cross section does NOT depend on the parameters (and would be the same for all
                              benchmarks). In this case, no reweighting is run.
        :param only_prepare_script: bool. If True, MadGraph is not executed, but instead a run.sh script is created in the
                                    process directory.
        :param initial_command: Initial shell commands that have to be executed before MG is run (e.g. loading a virtual
                                environment).
        :param log_file: Path to a log file in which the MadGraph output is saved.
        """

        if only_prepare_script:
            logging.info('Preparing script to run MadGraph and Pythia in %s', mg_process_directory)
        else:
            logging.info('Starting MadGraph and Pythia in %s', mg_process_directory)

        create_missing_folders([mg_process_directory, os.path.dirname(log_file)])
        if proc_card_filename is not None:
            create_missing_folders([proc_card_filename])

        return run_mg_pythia(
            mg_directory,
            mg_process_directory,
            proc_card_filename,
            run_card_file,
            param_card_file,
            reweight_card_file,
            pythia8_card_file,
            is_background=is_background,
            only_prepare_script=only_prepare_script,
            initial_command=initial_command,
            log_file=log_file
        )

    def run(self,
            mg_directory,
            proc_card_file,
            param_card_template_file,
            reweight_card_template_file,
            run_card_file=None,
            pythia8_card_file=None,
            mg_process_directory=None,
            ufo_model_directory=None,
            temp_directory=None,
            sample_benchmark=None,
            only_prepare_script=False,
            is_background=False,
            initial_command=None,
            log_directory=None):

        """
        Runs the event generation with MadGraph and Pythia.

        :param param_card_template_file:
        :param reweight_card_template_file:
        :param sample_benchmark:
        :param mg_directory: Path to the MadGraph 5 base directory.
        :param proc_card_file: Path to the process card that tells MadGraph how to generate the process.
        :param temp_directory: Path to a temporary directory.
        :param run_card_file: Path to the MadGraph run card. If None, the card present in the process folder is used.
        :param pythia8_card_file: Path to the MadGraph Pythia8 card. If None, the card present in the process folder
                                  is used.
        :param mg_process_directory: Path to the MG process directory. If None, use
                                     <MadGraph directory>/MadMiner_process.
        :param is_background: bool that should be True for background processes, i.e. process in which the differential
                              cross section does NOT depend on the parameters (and would be the same for all
                              benchmarks). In this case, no reweighting is run.
        :param only_prepare_script: bool. If True, MadGraph is not executed, but instead a run.sh script is created in the
                                    process directory.
        :param initial_command: Initial shell commands that have to be executed before MG is run (e.g. loading a virtual
                                environment).
        :param log_directory: Directory for log files with the MadGraph output.
        """

        if mg_process_directory is None:
            mg_process_directory = './MG_process'

        if temp_directory is None:
            temp_directory = tempfile.gettempdir()

        if log_directory is None:
            log_directory = './logs'

        log_file_generate = log_directory + '/generate.log'
        log_file_run = log_directory + '/run.log'

        self.generate_mg_process(mg_directory,
                                 temp_directory,
                                 proc_card_file,
                                 mg_process_directory,
                                 ufo_model_directory=ufo_model_directory,
                                 initial_command=initial_command,
                                 log_file=log_file_generate)

        self.export_cards(param_card_template_file,
                          reweight_card_template_file,
                          mg_process_directory,
                          sample_benchmark)

        self.run_mg_pythia(mg_directory,
                           mg_process_directory,
                           None,
                           run_card_file,
                           None,
                           None,
                           pythia8_card_file,
                           is_background=is_background,
                           only_prepare_script=only_prepare_script,
                           initial_command=initial_command,
                           log_file=log_file_run)

    def run_multiple(self,
                     mg_directory,
                     proc_card_file,
                     param_card_template_file,
                     reweight_card_template_file,
                     run_card_files=None,
                     pythia8_card_file=None,
                     mg_process_directory=None,
                     ufo_model_directory=None,
                     temp_directory=None,
                     sample_benchmarks=None,
                     is_background=False,
                     only_prepare_script=False,
                     initial_command=None,
                     log_directory=None):

        """
        Runs multiple event generation jobs with different run cards and / or importance sampling with MadGraph and Pythia.

        :param param_card_template_file:
        :param reweight_card_template_file:
        :param sample_benchmarks:
        :param mg_directory: Path to the MadGraph 5 base directory.
        :param proc_card_file: Path to the process card that tells MadGraph how to generate the process.
        :param temp_directory: Path to a temporary directory.
        :param run_card_files: List of paths to the MadGraph run card. If None, the card present in the process folder is used.
        :param pythia8_card_file: Path to the MadGraph Pythia8 card. If None, the card present in the process folder
                                  is used.
        :param mg_process_directory: Path to the MG process directory. If None, use
                                     <MadGraph directory>/MadMiner_process.
        :param is_background: bool that should be True for background processes, i.e. process in which the differential
                              cross section does NOT depend on the parameters (and would be the same for all
                              benchmarks). In this case, no reweighting is run.
        :param only_prepare_script: bool. If True, MadGraph is not executed, but instead a run.sh script is created in the
                                    process directory.
        :param initial_command: Initial shell commands that have to be executed before MG is run (e.g. loading a virtual
                                environment).
        :param log_directory: Directory for log files with the MadGraph output.
        """

        if mg_process_directory is None:
            mg_process_directory = './MG_process'

        if temp_directory is None:
            temp_directory = tempfile.gettempdir()

        if log_directory is None:
            log_directory = './logs'

        # Generate process folder
        log_file_generate = log_directory + '/generate.log'

        self.generate_mg_process(
            mg_directory,
            temp_directory,
            proc_card_file,
            mg_process_directory,
            ufo_model_directory=ufo_model_directory,
            initial_command=initial_command,
            log_file=log_file_generate
        )

        # Loop over settings
        i = 0
        results = []

        for run_card_file in run_card_files:
            for sample_benchmark in sample_benchmarks:

                log_file_run = log_directory + '/run_{}.log'.format(i)

                logging.info('Run %s', i)
                logging.info('  Run card:                %s', run_card_file)
                logging.info('  Sampling from benchmark: %s', sample_benchmark)
                logging.info('  Log file:                %s', log_file_run)

                self.export_cards(
                    param_card_template_file,
                    reweight_card_template_file,
                    mg_process_directory,
                    sample_benchmark
                )

                result = self.run_mg_pythia(
                    mg_directory,
                    mg_process_directory,
                    None,
                    run_card_file,
                    None,
                    None,
                    pythia8_card_file,
                    is_background=is_background,
                    only_prepare_script=only_prepare_script,
                    initial_command=initial_command,
                    log_file=log_file_run
                )

                results.append(result)

                i += 1

        # Master shell script
        if only_prepare_script:
            master_script_filename = mg_process_directory + '/madminer_run_all.sh'

            commands = '\n'.join(results)
            script = '#!/bin/bash\n\n{}'.format(commands)

            with open(master_script_filename, 'w') as file:
                file.write(script)

            logging.info('To run MadGraph, please execute %s', master_script_filename)
