class MadMiner():

    def __init__(self):

        """ Constructor """

        self.parameters = {}
        self.benchmarks = {}
        self.default_benchmark = None

    def add_parameter(self,
                      lha_block,
                      lha_id,
                      parameter_name=None):

        """ Adds an individual parameter """

        # Default names
        if parameter_name is None:
            parameter_name = 'param' + str(len(self.parameters))

        # Check input
        assert isinstance(parameter_name, str), 'Parameter name is not a string: {}'.format(parameter_name)
        assert isinstance(lha_block, str), 'LHA block is not a string: {}'.format(lha_block)
        assert isinstance(lha_id, int), 'LHA id is not an integer: {}'.format(lha_id)

        assert parameter_name not in self.parameters, 'Parameter name exists already: {}'.format(parameter_name)

        # Add parameter
        self.parameters[parameter_name] = (lha_block, lha_id)

    def set_parameters(self,
                       parameters):

        """ Defines whole parameter space

        :type parameters: dict or list that specifies all parameters. If dict, keys are the parameter names and values
                          are tuples (LHA block, LHA id). If list, entries are tuples (LHA block, LHA id) (the parameter
                          names are chosen automatically).
        """

        self.parameters = {}

        if isinstance(parameters, dict):
            for name, values in parameters.items():
                assert len(values) == 2, 'Parameter dict entry does not have length 2: {0}'.format(values)
                self.add_parameter(values[0], values[1], name)

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
            benchmark_name = 'benchmark' + str(len(self.benchmarks))

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
                       benchmarks):
        """
        Sets all parameter benchmarks.

        :param benchmarks: dict or list that specifies all parameters. If dict, the keys are the benchmark names and
                           the values are dicts {parameter_name:value}. If list, the entries are dicts
                           {parameter_name:value} (and the benchmark names are chosen automatically).
        """

        self.benchmarks = {}

        if isinstance(benchmarks, dict):
            for name, values in benchmarks.items():
                self.add_benchmark(values, name)
        else:
            for values in benchmarks:
                self.add_benchmark(values)

    def export_cards(self,
                     param_card_template_file,
                     reweight_card_template_file,
                     mg_process_directory,
                     sample_benchmark=None):

        """ Writes out param_card and reweight_card for MadGraph """

        # Check status
        assert self.default_benchmark is not None
        assert len(self.benchmarks) > 0

        # Default benchmark
        if sample_benchmark is None:
            sample_benchmark = self.default_benchmark

        # Parameters for param_card
        benchmark = self.benchmarks[sample_benchmark]

        # Open parameter card template
        with open(param_card_template_file) as file:
            param_card = file.read()

        # Replace parameter values
        for parameter_name, parameter_value in benchmark.keys():
            parameter_lha_block = self.parameters[parameter_name][0]
            parameter_lha_id = self.parameters[parameter_name][1]

            block_begin = param_card.find('Block ' + parameter_lha_block)
            if block_begin < 0:
                raise ValueError('Could not find block {0} in param_card template!'.format(parameter_lha_block))

            block_end = param_card.find('Block', beg=block_begin)
            if block_end < 0:
                block_end = len(param_card)

            block = param_card[block_begin:block_end].split('\n')
            changed_line = False
            for i, line in enumerate(block):
                comment_pos = line.find('#')
                if i >= 0:
                    line = line[:comment_pos]
                line = line.strip()
                elements = line.split()
                if len(elements) >= 2:
                    if elements[0] == parameter_lha_id:
                        block[i] = str(parameter_lha_id) + '    ' + str(parameter_value) + '    # MadMiner'
                        changed_line = True
                        break

            if not changed_line:
                raise ValueError('Could not find LHA ID {0} in param_card template!'.format(parameter_lha_id))

            param_card = param_card[:block_begin] + '\n'.join(block) + param_card[block_end:]

        # Save param_card.dat
        with open(mg_process_directory + '/Cards/param_card.dat') as file:
            file.write(param_card)

        # Open reweight_card template
        with open(reweight_card_template_file) as file:
            reweight_card = file.read()

        # Put in parameter values
        block_end = reweight_card.search('# Manual')
        assert block_end >= 0, 'Cannot find "# Manual" string in reweight_card template'

        insert_pos = reweight_card.rsearch('\n\n')
        assert insert_pos >= 0, 'Cannot find empty line in reweight_card template'

        lines = []
        for benchmark_name, benchmark in self.benchmarks:
            if benchmark_name == sample_benchmark:
                continue

            lines.append('')
            lines.append('# MadMiner benchmark ' + benchmark_name)
            lines.append('launch')

            for parameter_name, parameter_value in benchmark.keys():
                parameter_lha_block = self.parameters[parameter_name][0]
                parameter_lha_id = self.parameters[parameter_name][1]

                lines.append('  set {0} {1} {2}')

            lines.append('')

            reweight_card = reweight_card[:insert_pos] + '\n'.join(lines) + reweight_card[insert_pos:]

        # Save param_card.dat
        with open(mg_process_directory + '/Cards/reweight_card.dat') as file:
            file.write(reweight_card)

