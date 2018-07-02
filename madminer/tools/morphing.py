from __future__ import absolute_import, division, print_function, unicode_literals
import six

import numpy as np
from collections import OrderedDict


class Morpher:

    def __init__(self,
                 parameters_from_madminer=None,
                 parameter_max_power=None,
                 parameter_range=None,
                 fixed_benchmarks=None,
                 max_overall_power=4,
                 n_bases=1):

        """
        Constructor.

        :param parameters_from_madminer:
        :param parameter_max_power:
        :param parameter_range:
        :param fixed_benchmarks:
        :param max_overall_power:
        :param n_bases:
        """

        # Input
        self.n_bases = n_bases
        self.max_overall_power = max_overall_power

        # MadMiner interface
        if parameters_from_madminer is not None:
            self.use_madminer_interface = True
            self.n_parameters = len(parameters_from_madminer)
            self.parameter_names = [key for key in parameters_from_madminer]
            self.parameter_max_power = np.array(
                [parameters_from_madminer[key][2] for key in self.parameter_names],
                dtype=np.int
            )
            self.parameter_range = np.array(
                [parameters_from_madminer[key][3] for key in self.parameter_names]
            )

            if fixed_benchmarks is None:
                fixed_benchmarks = OrderedDict()
            self.fixed_benchmarks = []
            self.fixed_benchmark_names = []
            for bname, benchmark_in in six.iteritems(fixed_benchmarks):
                self.fixed_benchmark_names.append(bname)
                self.fixed_benchmarks.append(
                    [benchmark_in[key] for key in self.parameter_names]
                )
            self.fixed_benchmarks = np.array(self.fixed_benchmarks)

        # Generic interface
        else:
            self.use_madminer_interface = False
            self.n_parameters = len(parameter_max_power)
            self.parameter_names = None
            self.parameter_max_power = np.array(parameter_max_power, dtype=np.int)
            self.parameter_range = np.array(parameter_range)

            if fixed_benchmarks is None:
                fixed_benchmarks = np.array([])
            self.fixed_benchmarks = np.array(fixed_benchmarks)
            self.fixed_benchmark_names = []

        # Components
        self.components = self._find_components()
        self.n_components = len(self.components)
        self.n_benchmarks = self.n_bases * self.n_components
        self.n_missing_benchmarks = self.n_benchmarks - len(self.fixed_benchmarks)

        assert self.n_missing_benchmarks >= 0, 'Too many fixed benchmarks!'

        # Current chosen basis
        self.basis = None
        self.morphing_matrix = None

    def find_basis_simple(self,
                          n_trials=100,
                          n_test_thetas=100,
                          return_morphing_matrix=False):

        """
        Finds a set of basis parameter vectors, based on a rudimentary optimization algorithm.

        :param n_trials:
        :param n_test_thetas:
        :param return_morphing_matrix:
        :return:
        """

        # Optimization
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

        self.basis = best_basis
        self.morphing_matrix = best_morphing_matrix

        # Export to MadMiner
        if self.use_madminer_interface:
            basis_madminer = OrderedDict()
            for i, benchmark in enumerate(best_basis):
                if i < len(self.fixed_benchmark_names):
                    benchmark_name = self.fixed_benchmark_names[i]
                else:
                    benchmark_name = 'morphing_basis_vector_' + str(len(basis_madminer))
                parameter = OrderedDict()
                for p, pname in enumerate(self.parameter_names):
                    parameter[pname] = benchmark[p]
                basis_madminer[benchmark_name] = parameter

            if return_morphing_matrix:
                return basis_madminer, self.components, best_morphing_matrix
            return basis_madminer

        # Normal return
        if return_morphing_matrix:
            return best_basis, self.components, best_morphing_matrix
        return best_basis

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

        """ Propose a random basis """

        if len(self.fixed_benchmarks) > 0:
            basis = np.vstack([
                self.fixed_benchmarks,
                self._draw_random_thetas(self.n_missing_benchmarks)
            ])
        else:
            basis = self._draw_random_thetas(self.n_missing_benchmarks)

        return basis

    def _draw_random_thetas(self, n_thetas):

        """ Randomly draws parameter vectors within the specified parameter ranges """

        # First draw random numbers in range [0, 1)^n_parameters
        u = np.random.rand(n_thetas, self.n_parameters)

        # Transform to right range
        basis = self.parameter_range[:, 0] + u * (self.parameter_range[:, 1] - self.parameter_range[:, 0])

        return basis

    def _calculate_morphing_matrix(self, basis=None):

        """ Calculates the morphing matrix that links the components to the basis parameter vectors """

        if basis is None:
            basis = self.basis

        # Full morphing matrix. Will have shape (n_components, n_benchmarks) (note transposition later)
        morphing_matrix = np.zeros((self.n_benchmarks, self.n_components))

        # Morphing submatrix for each basis
        for i in range(self.n_bases):
            n_benchmarks_this_basis = self.n_components
            this_basis = basis[i * n_benchmarks_this_basis:(i + 1) * n_benchmarks_this_basis]

            inv_morphing_submatrix = np.zeros((n_benchmarks_this_basis, self.n_components))

            for b in range(n_benchmarks_this_basis):
                for c in range(self.n_components):
                    factor = 1.
                    for p in range(self.n_parameters):
                        factor *= float(this_basis[b, p] ** self.components[c, p])
                    inv_morphing_submatrix[b, c] = factor

            # Invert -? components expressed in basis points. Shape (n_components, n_benchmarks_this_basis)
            morphing_submatrix = np.linalg.inv(inv_morphing_submatrix)

            # For now, just use 1 / n_bases times the weights of each basis
            morphing_submatrix = morphing_submatrix / float(self.n_bases)

            # Write into full morphing matrix
            morphing_submatrix = morphing_submatrix.T
            morphing_matrix[i * n_benchmarks_this_basis:(i + 1) * n_benchmarks_this_basis] = morphing_submatrix

        morphing_matrix = morphing_matrix.T

        return morphing_matrix

    def _calculate_morphing_weights(self, theta, basis=None, morphing_matrix=None):

        """ Calculates the morphing weights w_b(theta) for a given basis {theta_b} """

        if basis is None:
            basis = self.basis
            morphing_matrix = self.morphing_matrix

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
        weights = morphing_matrix.T.dot(component_weights)

        return weights

    def _evaluate_morphing(self, basis=None, morphing_matrix=None, n_test_thetas=100):

        """ Evaluates an objective function for a given basis """

        if basis is None:
            basis = self.basis
            morphing_matrix = self.morphing_matrix

        if morphing_matrix is None:
            morphing_matrix = self._calculate_morphing_matrix(basis)

        thetas_test = self._draw_random_thetas(n_thetas=n_test_thetas)
        squared_weights = 0.

        for theta in thetas_test:
            weights = self._calculate_morphing_weights(theta, basis, morphing_matrix)
            squared_weights += np.sum(weights * weights)

        squared_weights /= float(n_test_thetas)

        return -squared_weights
