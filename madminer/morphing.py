import numpy as np


class MadMorpher():

    def __init__(self,
                 parameters,
                 max_overall_power=4,
                 n_bases=1):

        if n_bases != 1:
            raise NotImplementedError('Basis oversampling (n_bases > 1) is not supported yet')

        self.parameters = parameters
        self.n_parameters = len(parameters)
        self.n_bases = n_bases
        self.max_overall_power = max_overall_power

        # For the morphing, we need an order
        self.parameter_names = [key for key in self.parameters]
        self.parameter_max_power = np.array(
            [self.parameters[key][2] for key in self.parameter_names],
            dtype=np.int
        )
        self.parameter_range = np.array(
            [self.parameters[key][3] for key in self.parameter_names]
        )

        # Components
        self.components = self._find_components()
        self.n_components = len(self.components)
        self.n_benchmarks = self.n_bases * self.n_components

    def find_basis_simple(self,
                          n_trials=100,
                          return_morphing_matrix=False):

        best_basis = None
        best_morphing_matrix = None
        best_performance = None

        for i in range(n_trials):
            basis = self._draw_random_basis()
            morphing_matrix = self._calculate_morphing_matrix(basis)
            performance = self._evaluate_morphing(basis, morphing_matrix)

            if (best_performance is None
                    or best_basis is None
                    or best_morphing_matrix is None
                    or performance > best_performance):
                best_performance = performance
                best_basis = basis
                best_morphing_matrix = morphing_matrix

        if return_morphing_matrix:
            return best_basis, best_morphing_matrix

        # Export as nested dict
        basis_export = {}
        for benchmark in best_basis:
            benchmark_name = 'benchmark' + str(len(basis_export))
            parameter = {}
            for p, pname in enumerate(self.parameter_names):
                parameter[pname] = benchmark[p]
            basis_export[benchmark_name] = parameter

        return basis_export

    def _find_components(self):
        # Find all components with their parameter powers
        c = 0
        powers = np.zeros(self.n_parameters, dtype=np.int16)
        components = []
        continue_loop = True

        while continue_loop:
            components.append(np.copy(powers))

            # next setting
            c += 1

            # if we are below max_power in total, increase rightest digit
            if sum(powers) < self.max_overall_power:
                powers[self.n_parameters - 1] += 1

            # if we are at max_power, set to zero from the right and increase left neighbour
            else:
                continue_loop = False
                for pos in range(self.n_parameters - 1, 0, -1):
                    if powers[pos] > 0:
                        continue_loop = True
                        powers[pos] = 0
                        powers[pos - 1] += 1
                        break

            # go through individual digits and check self.operator_maxpowers
            for pos in range(self.n_parameters - 1, 0, -1):
                if powers[pos] > self.parameter_max_power[pos]:
                    powers[pos] = 0
                    powers[pos - 1] += 1

        return components

    def _draw_random_basis(self):
        # First draw random numbers in range [0, 1)^n_parameters
        u = np.random.rand((self.n_benchmarks, self.n_parameters))

        # Transform to right range
        basis = self.parameter_range[:, 0] + u * (self.parameter_range[:, 1] - self.parameter_range[:, 0])
        
        return basis

    def _calculate_morphing_matrix(self, basis):
        # Basis points expressed in components
        inv_morphing_matrix = np.zeros((self.n_benchmarks, self.n_components))

        for b in range(self.n_benchmarks):
            for c in range(self.n_components):
                factor = 1.
                for p in range(self.n_parameters):
                    factor *= float(basis[b, p] ** self.components[1 + c, p])
                self.dictionary_sample_component[b, c] = factor

        # Invert
        # Components expressed in basis points. Shape (n_components, n_benchmarks)
        # TODO: oversampling
        morphing_matrix = np.linalg.inv(inv_morphing_matrix)

        return morphing_matrix

    def _calculate_morphing_weights(self, theta, basis, morphing_matrix=None):

        if morphing_matrix is None:
            morphing_matrix = self._calculate_morphing_matrix(basis)

        # Calculate component weights
        component_weights = np.zeros(self.n_parameters)
        for c in range(self.n_components):
            factor = 1.
            for p in range(self.n_parameters):
                factor *= float(theta[p] ** self.components[1 + c, p])
            component_weights[c] = factor
        component_weights = np.array(component_weights)

        # Transform to basis weights
        weights = morphing_matrix.T.dot(component_weights)  # TODO: Cross-check

        return weights

    def _evaluate_morphing(self, basis, morphing_matrix):
        pass

        ####################################################################################################################
        ####################################################################################################################
        ####################################################################################################################

        # Settings
        self.theta_unit_step = 1.64949627e-05  # 1 / v^2 in GeV^-2

        ############################################################
        # Components
        ############################################################

        # First, build list of all components
        component = 0
        powers = np.zeros((self.n_operators), dtype=np.int16)
        higgs_component_power_list = []
        continue_loop = True

        while continue_loop:
            higgs_component_power_list.append(np.copy(powers))

            # next setting
            component += 1

            # if we are below max_power in total, increase rightest digit
            if sum(powers) < self.total_maxpower:
                powers[self.n_operators - 1] += 1

            # if we are at max_power, set to zero from the right and increase left neighbour
            else:
                continue_loop = False
                for pos in range(self.n_operators - 1, 0, -1):
                    if powers[pos] > 0:
                        continue_loop = True
                        powers[pos] = 0
                        powers[pos - 1] += 1
                        break

            # go through individual digits and check self.operator_maxpowers
            for pos in range(self.n_operators - 1, 0, -1):
                if powers[pos] > self.operator_maxpowers[pos]:
                    powers[pos] = 0
                    powers[pos - 1] += 1

        # Now organize the results a little
        self.n_components_higgs = len(higgs_component_power_list)
        self.n_components = self.n_components_higgs + 1
        self.dictionary_component_power = np.zeros((self.n_components, self.n_operators), dtype=np.int16)

        if self.verbose:
            print
            ''
            print
            self.n_operators, 'operators'
            print
            self.n_components_higgs, 'Higgs components'

        for i in range(self.n_components_higgs):
            self.dictionary_component_power[i + 1] = higgs_component_power_list[i]

        if self.verbose:
            print
            ''
            print
            'Components with their operator powers:'
            for i, line in enumerate(self.dictionary_component_power):
                print
                i, '-', line

        self.sm_component_nonhiggs = 0
        self.sm_component_higgs = 1

        #################################################################################
        # Define samples
        #################################################################################

        self.n_samples = self.n_components_higgs

        # Just use the component matrix and rescale it such that the maximum is at one
        self.dictionary_sample_parameter = self.dictionary_component_power[1:, :] * (1. / float(self.total_maxpower))

        # Integration channels: only SM for now
        self.sample_integrate_channel = [False for s in range(self.n_samples)]
        self.sample_integrate_channel[0] = True

        if self.verbose:
            print
            ''
            print
            'Samples with their parameter points:'
            # print self.dictionary_sample_parameter
            for i, line in enumerate(self.dictionary_sample_parameter):
                print
                i, '-', line

        #################################################################################
        # Link samples and components
        #################################################################################

        # Dictionaries
        self.dictionary_sample_component = np.zeros((self.n_samples, self.n_components_higgs))
        self.dictionary_component_sample = np.zeros((self.n_components, self.n_samples))

        for s in range(self.n_samples):
            for c in range(self.n_components_higgs):
                factor = 1.
                for o in range(self.n_operators):
                    factor *= float(self.dictionary_sample_parameter[s, o] ** self.dictionary_component_power[1 + c, o])
                self.dictionary_sample_component[s, c] = factor

        if self.verbose:
            print
            ''
            print
            'Samples separated into components:'
            print
            self.dictionary_sample_component

        # Invert
        self.dictionary_component_sample[1:, :] = np.linalg.inv(self.dictionary_sample_component)

        if self.verbose:
            print
            ''
            print
            'Components separated into samples:'
            print
            self.dictionary_component_sample
