class MadMiner():

    def __init__(self):
        pass


    def set_parameters(self,
                       parameters):

        """ Defines whole parameter space """

        raise NotImplementedError


    def add_parameter(self,
                      lha_block,
                      lha_id,
                      parameter_name=None):

        """ Adds an individual parameter """

        raise NotImplementedError


    def set_benchmarks(self,
                       benchmarks):

        """ Defines all parameter benchmarks """

        raise NotImplementedError


    def add_benchmark(self,
                      parameter_values,
                      benchmark_name=None,
                      sample='auto'):

        """ Adds an individual parameter benchmark """

        raise NotImplementedError


    def export_cards(self,
                     param_card_template_file,
                     folder):

        """ Writes out param_card and reweight_card for MadGraph """

        raise NotImplementedError