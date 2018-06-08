from collections import OrderedDict

from .morphing import MadMorpher
from .mg_interface import export_param_card, export_reweight_card


class MadMiner:

    def __init__(self):

        """ Constructor """

        self.parameters = OrderedDict()
        self.benchmarks = OrderedDict()
        self.default_benchmark = None
        self.current_morpher = None

    def add_parameter(self,
                      lha_block,
                      lha_id,
                      parameter_name=None,
                      morphing_max_power=2,
                      morphing_parameter_range=(0.,1.)):

        """ Adds an individual parameter

        :param lha_block: str, the name of the LHA block as used in the param_card. Case-sensitive.
        :param lha_id: int, the LHA id as used in the param_card.
        :param parameter_name: an internal name for the parameter. If None, a the default 'benchmark_i' is used.
        :param morphing_max_power: the maximal power with which this parameter contributes to the squared matrix element
                                   of the process of interest. Typically at tree level, this is 2 for parameters that
                                   affect one vertex (e.g. only production or only decay of a particle), and 4 for
                                   parameters that affect two vertices (e.g. production and decay).
        :param morphing_parameter_range: tuple, the range of parameter values of primary interest. Only affects the
                                         basis optimization.
        """

        # Default names
        if parameter_name is None:
            parameter_name = 'parameter_' + str(len(self.parameters))

        # Check and sanitize input
        assert isinstance(parameter_name, str), 'Parameter name is not a string: {}'.format(parameter_name)
        assert isinstance(lha_block, str), 'LHA block is not a string: {}'.format(lha_block)
        assert isinstance(lha_id, int), 'LHA id is not an integer: {}'.format(lha_id)

        parameter_name = parameter_name.replace(' ', '_')
        parameter_name = parameter_name.replace('-', '_')

        assert parameter_name not in self.parameters, 'Parameter name exists already: {}'.format(parameter_name)

        # Add parameter
        self.parameters[parameter_name] = (lha_block, lha_id, morphing_max_power, morphing_parameter_range)

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
            for key, values in parameters.items():
                if len(values) == 5:
                    self.add_parameter(
                        lha_block=values[0],
                        lha_id=values[1],
                        parameter_name=key,
                        morphing_parameter_range=[values[3], values[4]],
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

        for key, value in parameter_values.items():
            assert key in self.parameters, 'Unknown parameter: {0}'.format(key)

        assert benchmark_name not in self.benchmarks, 'Benchmark name exists already: {}'.format(benchmark_name)

        # Add benchmark
        self.benchmarks[benchmark_name] = parameter_values

        # If first benchmark, this will be the default for sampling
        if len(self.benchmarks) == 1:
            self.default_benchmark = benchmark_name

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
            for name, values in benchmarks.items():
                self.add_benchmark(values, name)
        else:
            for values in benchmarks:
                self.add_benchmark(values)

    def set_benchmarks_from_morphing(self,
                                     max_overall_power=4,
                                     n_bases=1,
                                     keep_existing_benchmarks=False,
                                     n_trials=100,
                                     n_test_thetas=100):
        """
        Sets all parameter benchmarks based on a morphing algorithm. The morphing basis is optimized with respect to the
        expected mean squared morphing weights over the parameter region of interest.

        :param max_overall_power: The maximal sum of powers of all parameters.
        :param n_bases: The number of morphing bases generated. If n_bases > 1, multiple bases are combined, and the
                        weights for each basis are reduced by a factor 1 / n_bases. Currently not supported.
        :param keep_existing_benchmarks: Whether the previously defined benchmarks are included in the basis. In that
                                         case, the number of free parameters in the optimization routine is reduced.
        :param n_trials: Number of candidate bases used in the optimization.
        :param n_test_thetas: Number of validation benchmark points used to calculate the expected mean squared morphing
                              weight.
        """
        if keep_existing_benchmarks:
            morpher = MadMorpher(self.parameters,
                                 self.benchmarks,
                                 max_overall_power=max_overall_power,
                                 n_bases=n_bases)
        else:
            morpher = MadMorpher(self.parameters,
                                 max_overall_power=max_overall_power,
                                 n_bases=n_bases)

        self.current_morpher = morpher

        basis = morpher.find_basis_simple(n_trials=n_trials,
                                          n_test_thetas=n_test_thetas)
        self.set_benchmarks(basis)

    def set_benchmarks_from_finite_differences(self):
        raise NotImplementedError

    def export_cards(self,
                     param_card_template_file,
                     reweight_card_template_file,
                     mg_process_directory,
                     sample_benchmark=None):

        """
        Writes out param_card and reweight_card for MadGraph. Currently, this is the final output of this package.
        Future versions are scheduled to support the automatic running of MadGraph, and the automated conversion of event
        files.

        :param param_card_template_file: Path to a param_card.dat of the used model.
        :param reweight_card_template_file: Path to an empty reweight_card.dat (no commands, the default 'launch' should
                                            be commented out or removed).
        :param mg_process_directory: Path to the directory of the MG process.
        :param sample_benchmark: Name of the benchmark used for sampling. If None, the very first defined benchmark is
                                 used.
        """

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
