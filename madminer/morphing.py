import numpy as np
from collections import OrderedDict


class MadMorpher():

    def __init__(self,
                 parameters,
                 fixed_benchmarks=None,
                 max_overall_power=4,
                 n_bases=1):

        """
        Constructor.

        :param parameters:
        :param max_overall_power:
        :param n_bases:
        """

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

        # External fixed benchmarks
        if fixed_benchmarks is None:
            fixed_benchmarks = OrderedDict()

        self.fixed_benchmarks = []
        for _, benchmark_in in fixed_benchmarks.items():
            self.fixed_benchmarks.append(
                [benchmark_in[key] for key in self.parameter_names]
            )
        self.fixed_benchmarks = np.array(self.fixed_benchmarks)

        # Components
        self.components = self._find_components()
        self.n_components = len(self.components)
        self.n_benchmarks = self.n_bases * self.n_components
        self.n_missing_benchmarks = self.n_benchmarks - len(self.fixed_benchmarks)

        assert(self.n_missing_benchmarks >= 0, 'Too many fixed benchmarks!')

        # Current chosen basis
        self.current_basis = None

    def find_basis_simple(self,
                          n_trials=100,
                          n_test_thetas=100,
                          return_morphing_matrix=False):

        """
        Finds a set of basis parameter vectors, based on a rudimentary optimization algorithm.

        :param n_trials:
        :param return_morphing_matrix:
        :return:
        """

        best_basis = None
        best_morphing_matrix = None
        best_performance = None

        for i in range(n_trials):
            basis = self._propose_basis()
            morphing_matrix = self._calculate_morphing_matrix(basis)
            performance = self._evaluate_morphing(basis, morphing_matrix, n_test_thetas=n_test_thetas)

            if (best_performance is None
                    or best_basis is None
                    or best_morphing_matrix is None
                    or performance > best_performance):
                best_performance = performance
                best_basis = basis
                best_morphing_matrix = morphing_matrix

        self.current_basis = best_basis

        # Export as nested dict
        basis_export = OrderedDict()
        for benchmark in best_basis:
            benchmark_name = 'morphing_basis_vector_' + str(len(basis_export))
            parameter = OrderedDict()
            for p, pname in enumerate(self.parameter_names):
                parameter[pname] = benchmark[p]
            basis_export[benchmark_name] = parameter

        if return_morphing_matrix:
            return basis_export, self.components, best_morphing_matrix
        return basis_export

    def _find_components(self):

        """ Find and return a list of components with their power dependence on the parameters """

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

        components = np.array(components, dtype=np.int)

        return components

    def _propose_basis(self):

        if len(self.fixed_benchmarks) > 0:
            basis = np.vstack([
                self.fixed_benchmarks,
                self._draw_random_thetas(self.n_missing_benchmarks)
            ])
        else:
            basis = self._draw_random_thetas(self.n_missing_benchmarks)

        return basis

    def _draw_random_thetas(self, n_thetas):

        """ Randomly draws basis vectors within the specified parameter ranges """

        # First draw random numbers in range [0, 1)^n_parameters
        u = np.random.rand(n_thetas, self.n_parameters)

        # Transform to right range
        basis = self.parameter_range[:, 0] + u * (self.parameter_range[:, 1] - self.parameter_range[:, 0])

        return basis

    def _calculate_morphing_matrix(self, basis=None):

        """ Calculates the morphing matrix that links the components to the basis parameter vectors """

        if basis is None:
            basis = self.current_basis

        # Basis points expressed in components
        inv_morphing_matrix = np.zeros((self.n_benchmarks, self.n_components))

        for b in range(self.n_benchmarks):
            for c in range(self.n_components):
                factor = 1.
                for p in range(self.n_parameters):
                    factor *= float(basis[b, p] ** self.components[c, p])
                inv_morphing_matrix[b, c] = factor

        # Invert
        # Components expressed in basis points. Shape (n_components, n_benchmarks)
        # TODO: oversampling
        morphing_matrix = np.linalg.inv(inv_morphing_matrix)

        return morphing_matrix

    def _calculate_morphing_weights(self, theta, basis=None, morphing_matrix=None):

        """ Calculates the morphing weights w_b(theta) for a given basis {theta_b} """

        if basis is None:
            basis = self.current_basis

        if morphing_matrix is None:
            morphing_matrix = self._calculate_morphing_matrix(basis)

        # Calculate component weights
        component_weights = np.zeros(self.n_components)
        for c in range(self.n_components):
            factor = 1.
            for p in range(self.n_parameters):
                factor *= float(theta[p] ** self.components[c, p])
            component_weights[c] = factor
        component_weights = np.array(component_weights)

        # Transform to basis weights
        weights = morphing_matrix.T.dot(component_weights)  # TODO: Cross-check

        return weights

    def _evaluate_morphing(self, basis=None, morphing_matrix=None, n_test_thetas=100):

        """ Evaluates an objective function for a given basis """

        if basis is None:
            basis = self.current_basis

        if morphing_matrix is None:
            morphing_matrix = self._calculate_morphing_matrix(basis)

        thetas_test = self._draw_random_thetas(n_thetas=n_test_thetas)
        squared_weights = 0.

        for theta in thetas_test:
            weights = self._calculate_morphing_weights(theta, basis, morphing_matrix)
            squared_weights += np.sum(weights * weights)

        squared_weights /= float(n_test_thetas)

        return - squared_weights
