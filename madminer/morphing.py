import numpy as np
from collections import OrderedDict


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
        basis_export = OrderedDict()
        for benchmark in best_basis:
            benchmark_name = 'benchmark' + str(len(basis_export))
            parameter = OrderedDict()
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
