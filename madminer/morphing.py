from __future__ import absolute_import, division, print_function, unicode_literals
import six

import logging
import numpy as np
from collections import OrderedDict
import itertools


class SimpleMorpher:
    """ """

    def __init__(self,
                 parameters_from_madminer=None,
                 parameter_max_power=None,
                 parameter_range=None):

        """
        Constructor.

        :param parameters_from_madminer:
        :param parameter_max_power:
        :param parameter_range:
        """

        # GoldMine interface
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

        # Generic interface
        else:
            self.use_madminer_interface = False
            self.n_parameters = len(parameter_max_power)
            self.parameter_names = None
            self.parameter_max_power = np.array(parameter_max_power, dtype=np.int)
            self.parameter_range = np.array(parameter_range)

        # Currently empty
        self.components = None
        self.n_components = None
        self.basis = None
        self.morphing_matrix = None

    def set_components(self,
                       components):
        """

        Parameters
        ----------
        components :
            

        Returns
        -------

        """

        self.components = components
        self.n_components = len(self.components)

    def find_components(self,
                        max_overall_power=4):

        """Find and return a list of components with their power dependence on the parameters

        Parameters
        ----------
        max_overall_power :
             (Default value = 4)

        Returns
        -------

        """

        components = []

        powers_each_component = [range(self.parameter_max_power[i] + 1) for i in range(self.n_parameters)]

        for powers in itertools.product(*powers_each_component):
            powers = np.array(powers, dtype=np.int)

            if np.sum(powers) > max_overall_power:
                continue

            components.append(np.copy(powers))

        self.components = np.array(components, dtype=np.int)
        self.n_components = len(self.components)

        return self.components

    def set_basis(self,
                  basis_from_madminer=None,
                  basis_numpy=None,
                  morphing_matrix=None):
        """

        Parameters
        ----------
        basis_from_madminer :
             (Default value = None)
        basis_numpy :
             (Default value = None)
        morphing_matrix :
             (Default value = None)

        Returns
        -------

        """

        if basis_from_madminer is not None:
            self.basis = []
            for bname, benchmark_in in six.iteritems(basis_from_madminer):
                self.basis.append(
                    [benchmark_in[key] for key in self.parameter_names]
                )
            self.basis = np.array(self.basis)
        elif basis_numpy is not None:
            self.basis = np.array(basis_numpy)
        else:
            raise RuntimeError('No basis given')

        if morphing_matrix is None:
            self.morphing_matrix = self.calculate_morphing_matrix()
        else:
            self.morphing_matrix = morphing_matrix

    def optimize_basis(self,
                       n_bases=1,
                       fixed_benchmarks_from_madminer=None,
                       fixed_benchmarks_numpy=None,
                       n_trials=100,
                       n_test_thetas=100):

        """Finds a set of basis parameter vectors, based on a rudimentary optimization algorithm.

        Parameters
        ----------
        fixed_benchmarks_numpy :
            param fixed_benchmarks_from_madminer: (Default value = None)
        n_bases :
            param n_trials: (Default value = 1)
        n_test_thetas :
            return: (Default value = 100)
        fixed_benchmarks_from_madminer :
             (Default value = None)
        n_trials :
             (Default value = 100)

        Returns
        -------

        """

        # Check all data is there
        if self.components is None or self.n_components is None or self.n_components <= 0:
            raise RuntimeError('No components defined. Use morpher.set_components() or morpher.find_components() '
                               'first!')

        # Fixed benchmarks
        if fixed_benchmarks_from_madminer is not None:
            fixed_benchmarks = []
            fixed_benchmark_names = []
            for bname, benchmark_in in six.iteritems(fixed_benchmarks_from_madminer):
                fixed_benchmark_names.append(bname)
                fixed_benchmarks.append(
                    [benchmark_in[key] for key in self.parameter_names]
                )
            fixed_benchmarks = np.array(fixed_benchmarks)
        elif fixed_benchmarks_numpy is not None:
            fixed_benchmarks = np.array(fixed_benchmarks_numpy)
            fixed_benchmark_names = []
        else:
            fixed_benchmarks = np.array([])
            fixed_benchmark_names = []

        # Missing benchmarks
        n_benchmarks = n_bases * self.n_components
        n_missing_benchmarks = n_benchmarks - len(fixed_benchmarks)

        assert n_missing_benchmarks >= 0, 'Too many fixed benchmarks!'

        # Optimization
        best_basis = None
        best_morphing_matrix = None
        best_performance = None

        for i in range(n_trials):
            basis = self._propose_basis(fixed_benchmarks, n_missing_benchmarks)
            morphing_matrix = self.calculate_morphing_matrix(basis)
            performance = self.evaluate_morphing(basis, morphing_matrix, n_test_thetas=n_test_thetas)

            if (best_performance is None
                    or best_basis is None
                    or best_morphing_matrix is None
                    or performance > best_performance):
                best_performance = performance
                best_basis = basis
                best_morphing_matrix = morphing_matrix

        # Save
        self.basis = best_basis
        self.morphing_matrix = best_morphing_matrix

        # GoldMine output
        if self.use_madminer_interface:
            basis_madminer = OrderedDict()
            for i, benchmark in enumerate(best_basis):
                if i < len(fixed_benchmark_names):
                    benchmark_name = fixed_benchmark_names[i]
                else:
                    benchmark_name = 'morphing_basis_vector_' + str(len(basis_madminer))
                parameter = OrderedDict()
                for p, pname in enumerate(self.parameter_names):
                    parameter[pname] = benchmark[p]
                basis_madminer[benchmark_name] = parameter

            return basis_madminer

        # Normal output
        return best_basis

    def _propose_basis(self, fixed_benchmarks, n_missing_benchmarks):

        """Propose a random basis

        Parameters
        ----------
        fixed_benchmarks :
            
        n_missing_benchmarks :
            

        Returns
        -------

        """

        if len(fixed_benchmarks) > 0:
            basis = np.vstack([
                fixed_benchmarks,
                self._draw_random_thetas(n_missing_benchmarks)
            ])
        else:
            basis = self._draw_random_thetas(n_missing_benchmarks)

        return basis

    def _draw_random_thetas(self, n_thetas):

        """Randomly draws parameter vectors within the specified parameter ranges

        Parameters
        ----------
        n_thetas :
            

        Returns
        -------

        """

        # First draw random numbers in range [0, 1)^n_parameters
        u = np.random.rand(n_thetas, self.n_parameters)

        # Transform to right range
        thetas = self.parameter_range[:, 0] + u * (self.parameter_range[:, 1] - self.parameter_range[:, 0])

        return thetas

    def calculate_morphing_matrix(self, basis=None):

        """Calculates the morphing matrix that links the components to the basis parameter vectors

        Parameters
        ----------
        basis :
             (Default value = None)

        Returns
        -------

        """

        # Check all data is there
        if self.components is None or self.n_components is None or self.n_components <= 0:
            raise RuntimeError('No components defined. Use morpher.set_components() or morpher.find_components() '
                               'first!')

        if basis is None:
            basis = self.basis

        if basis is None:
            raise RuntimeError('No basis defined or given. Use Morpher.set_basis(), Morpher.optimize_basis(), or the '
                               'basis keyword.')

        n_benchmarks = len(basis)
        n_bases = n_benchmarks // self.n_components
        assert n_bases * self.n_components == n_benchmarks, 'Basis and number of components incompatible!'

        # Full morphing matrix. Will have shape (n_components, n_benchmarks) (note transposition later)
        morphing_matrix = np.zeros((n_benchmarks, self.n_components))

        # Morphing submatrix for each basis
        for i in range(n_bases):
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
            morphing_submatrix = morphing_submatrix / float(n_bases)

            # Write into full morphing matrix
            morphing_submatrix = morphing_submatrix.T
            morphing_matrix[i * n_benchmarks_this_basis:(i + 1) * n_benchmarks_this_basis] = morphing_submatrix

        morphing_matrix = morphing_matrix.T

        return morphing_matrix

    def calculate_morphing_weights(self, theta, basis=None, morphing_matrix=None):

        """Calculates the morphing weights w_b(theta) for a given basis {theta_b}

        Parameters
        ----------
        theta :
            
        basis :
             (Default value = None)
        morphing_matrix :
             (Default value = None)

        Returns
        -------

        """

        # Check all data is there
        if self.components is None or self.n_components is None or self.n_components <= 0:
            raise RuntimeError('No components defined. Use morpher.set_components() or morpher.find_components() '
                               'first!')

        if basis is None:
            basis = self.basis
            morphing_matrix = self.morphing_matrix

        if basis is None:
            raise RuntimeError('No basis defined or given. Use Morpher.set_basis(), Morpher.optimize_basis(), or the '
                               'basis keyword.')

        if morphing_matrix is None:
            morphing_matrix = self.calculate_morphing_matrix(basis)

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

    def calculate_morphing_weight_gradient(self, theta, basis=None, morphing_matrix=None):

        """Calculates the gradient of the morphing weights grad_i w_b(theta). Output shape is (gradient
        direction, basis benchmarks).

        Parameters
        ----------
        theta :
            
        basis :
             (Default value = None)
        morphing_matrix :
             (Default value = None)

        Returns
        -------

        """

        # Check all data is there
        if self.components is None or self.n_components is None or self.n_components <= 0:
            raise RuntimeError('No components defined. Use morpher.set_components() or morpher.find_components() '
                               'first!')

        if basis is None:
            basis = self.basis
            morphing_matrix = self.morphing_matrix

        if basis is None:
            raise RuntimeError('No basis defined or given. Use Morpher.set_basis(), Morpher.optimize_basis(), or the '
                               'basis keyword.')

        if morphing_matrix is None:
            morphing_matrix = self.calculate_morphing_matrix(basis)

        # Calculate gradients of component weights wrt theta
        component_weight_gradients = np.zeros((self.n_components, self.n_parameters))

        for c in range(self.n_components):
            for i in range(self.n_parameters):
                factor = 1.
                for p in range(self.n_parameters):
                    if p == i and self.components[c, p] > 0:
                        factor *= float(self.components[c, p]) * theta[p] ** (self.components[c, p] - 1)
                    elif p == i:
                        factor = 0.
                        break
                    else:
                        factor *= float(theta[p] ** self.components[c, p])
                component_weight_gradients[c, i] = factor

        # Transform to basis weights
        weight_gradients = morphing_matrix.T.dot(component_weight_gradients).T  # Shape (n_parameters, n_benchmarks)

        return weight_gradients

    def evaluate_morphing(self, basis=None, morphing_matrix=None, n_test_thetas=100, return_weights_and_thetas=False):

        """Evaluates an objective function for a given basis

        Parameters
        ----------
        basis :
             (Default value = None)
        morphing_matrix :
             (Default value = None)
        n_test_thetas :
             (Default value = 100)
        return_weights_and_thetas :
             (Default value = False)

        Returns
        -------

        """

        # Check all data is there
        if self.components is None or self.n_components is None or self.n_components <= 0:
            raise RuntimeError('No components defined. Use morpher.set_components() or morpher.find_components() '
                               'first!')

        if basis is None:
            basis = self.basis
            morphing_matrix = self.morphing_matrix

        if basis is None:
            raise RuntimeError('No basis defined or given. Use Morpher.set_basis(), Morpher.optimize_basis(), or the '
                               'basis keyword.')

        if morphing_matrix is None:
            morphing_matrix = self.calculate_morphing_matrix(basis)

        thetas_test = self._draw_random_thetas(n_thetas=n_test_thetas)
        squared_weights = 0.
        squared_weight_list = []

        for theta in thetas_test:
            weights = self.calculate_morphing_weights(theta, basis, morphing_matrix)
            squared_weights += np.sum(weights * weights)

            if return_weights_and_thetas:
                squared_weight_list.append(np.sum(weights * weights))

        if return_weights_and_thetas:
            return thetas_test, np.array(squared_weight_list)

        squared_weights /= float(n_test_thetas)

        return -squared_weights


class AdvancedMorpher:
    """ """

    def __init__(self,
                 parameters_from_madminer=None,
                 parameter_max_power=None,
                 parameter_range=None):

        """
        Constructor.

        :param parameters_from_madminer:
        :param parameter_max_power:
        :param parameter_range:
        """

        # GoldMine interface
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

        # Generic interface
        else:
            self.use_madminer_interface = False
            self.n_parameters = len(parameter_max_power)
            self.parameter_names = None
            self.parameter_max_power = parameter_max_power
            self.parameter_range = np.array(parameter_range)

        # Currently empty
        self.components = None
        self.n_components = None
        self.basis = None
        self.morphing_matrix = None

    def set_components(self,
                       components):
        """

        Parameters
        ----------
        components :
            

        Returns
        -------

        """

        self.components = components
        self.n_components = len(self.components)

    def find_components(self,
                        max_overall_power=4):

        """Find and return a list of components with their power dependence on the parameters

        Parameters
        ----------
        max_overall_power :
             (Default value = 4)

        Returns
        -------

        """

        if isinstance(max_overall_power, int):
            max_overall_power = [max_overall_power]

        n_regions = len(max_overall_power)

        # Check that number of regions is consistent
        for max_power in self.parameter_max_power:
            if n_regions != len(max_power):
                raise RuntimeError('Parameters have different number of partitions of max powers: {} {}'.format(
                    max_overall_power, self.parameter_max_power))

        # Go through regions and find components for each
        components = []

        for i in range(n_regions):
            this_max_overall_power = max_overall_power[i]
            powers_each_component = [range(max_power[i] + 1) for max_power in self.parameter_max_power]

            logging.debug('Region %s: max overall power %s, max individual powers %s',
                          i, this_max_overall_power, [max_power[i] for max_power in self.parameter_max_power])

            for powers in itertools.product(*powers_each_component):
                powers = np.array(powers, dtype=np.int)

                if np.sum(powers) > this_max_overall_power:
                    continue

                if not any((powers == x).all() for x in components):
                    logging.debug('  Adding component %s', powers)
                    components.append(np.copy(powers))
                else:
                    logging.debug('  Not adding component %s again', powers)

        self.components = np.array(components, dtype=np.int)
        self.n_components = len(self.components)

        return self.components

    def set_basis(self,
                  basis_from_madminer=None,
                  basis_numpy=None,
                  morphing_matrix=None):
        """

        Parameters
        ----------
        basis_from_madminer :
             (Default value = None)
        basis_numpy :
             (Default value = None)
        morphing_matrix :
             (Default value = None)

        Returns
        -------

        """

        if basis_from_madminer is not None:
            self.basis = []
            for bname, benchmark_in in six.iteritems(basis_from_madminer):
                self.basis.append(
                    [benchmark_in[key] for key in self.parameter_names]
                )
            self.basis = np.array(self.basis)
        elif basis_numpy is not None:
            self.basis = np.array(basis_numpy)
        else:
            raise RuntimeError('No basis given')

        if morphing_matrix is None:
            self.morphing_matrix = self.calculate_morphing_matrix()
        else:
            self.morphing_matrix = morphing_matrix

    def optimize_basis(self,
                       n_bases=1,
                       fixed_benchmarks_from_madminer=None,
                       fixed_benchmarks_numpy=None,
                       n_trials=100,
                       n_test_thetas=100):

        """Finds a set of basis parameter vectors, based on a rudimentary optimization algorithm.

        Parameters
        ----------
        fixed_benchmarks_numpy :
            param fixed_benchmarks_from_madminer: (Default value = None)
        n_bases :
            param n_trials: (Default value = 1)
        n_test_thetas :
            return: (Default value = 100)
        fixed_benchmarks_from_madminer :
             (Default value = None)
        n_trials :
             (Default value = 100)

        Returns
        -------

        """

        # Check all data is there
        if self.components is None or self.n_components is None or self.n_components <= 0:
            raise RuntimeError('No components defined. Use morpher.set_components() or morpher.find_components() '
                               'first!')

        # Fixed benchmarks
        if fixed_benchmarks_from_madminer is not None:
            fixed_benchmarks = []
            fixed_benchmark_names = []
            for bname, benchmark_in in six.iteritems(fixed_benchmarks_from_madminer):
                fixed_benchmark_names.append(bname)
                fixed_benchmarks.append(
                    [benchmark_in[key] for key in self.parameter_names]
                )
            fixed_benchmarks = np.array(fixed_benchmarks)
        elif fixed_benchmarks_numpy is not None:
            fixed_benchmarks = np.array(fixed_benchmarks_numpy)
            fixed_benchmark_names = []
        else:
            fixed_benchmarks = np.array([])
            fixed_benchmark_names = []

        # Missing benchmarks
        n_benchmarks = n_bases * self.n_components
        n_missing_benchmarks = n_benchmarks - len(fixed_benchmarks)

        assert n_missing_benchmarks >= 0, 'Too many fixed benchmarks!'

        # Optimization
        best_basis = None
        best_morphing_matrix = None
        best_performance = None

        for i in range(n_trials):
            basis = self._propose_basis(fixed_benchmarks, n_missing_benchmarks)
            morphing_matrix = self.calculate_morphing_matrix(basis)
            performance = self.evaluate_morphing(basis, morphing_matrix, n_test_thetas=n_test_thetas)

            if (best_performance is None
                    or best_basis is None
                    or best_morphing_matrix is None
                    or performance > best_performance):
                best_performance = performance
                best_basis = basis
                best_morphing_matrix = morphing_matrix

        # Save
        self.basis = best_basis
        self.morphing_matrix = best_morphing_matrix

        # GoldMine output
        if self.use_madminer_interface:
            basis_madminer = OrderedDict()
            for i, benchmark in enumerate(best_basis):
                if i < len(fixed_benchmark_names):
                    benchmark_name = fixed_benchmark_names[i]
                else:
                    benchmark_name = 'morphing_basis_vector_' + str(len(basis_madminer))
                parameter = OrderedDict()
                for p, pname in enumerate(self.parameter_names):
                    parameter[pname] = benchmark[p]
                basis_madminer[benchmark_name] = parameter

            return basis_madminer

        # Normal output
        return best_basis

    def _propose_basis(self, fixed_benchmarks, n_missing_benchmarks):

        """Propose a random basis

        Parameters
        ----------
        fixed_benchmarks :
            
        n_missing_benchmarks :
            

        Returns
        -------

        """

        if len(fixed_benchmarks) > 0:
            basis = np.vstack([
                fixed_benchmarks,
                self._draw_random_thetas(n_missing_benchmarks)
            ])
        else:
            basis = self._draw_random_thetas(n_missing_benchmarks)

        return basis

    def _draw_random_thetas(self, n_thetas):

        """Randomly draws parameter vectors within the specified parameter ranges

        Parameters
        ----------
        n_thetas :
            

        Returns
        -------

        """

        # First draw random numbers in range [0, 1)^n_parameters
        u = np.random.rand(n_thetas, self.n_parameters)

        # Transform to right range
        thetas = self.parameter_range[:, 0] + u * (self.parameter_range[:, 1] - self.parameter_range[:, 0])

        return thetas

    def calculate_morphing_matrix(self, basis=None):

        """Calculates the morphing matrix that links the components to the basis parameter vectors

        Parameters
        ----------
        basis :
             (Default value = None)

        Returns
        -------

        """

        # Check all data is there
        if self.components is None or self.n_components is None or self.n_components <= 0:
            raise RuntimeError('No components defined. Use morpher.set_components() or morpher.find_components() '
                               'first!')

        if basis is None:
            basis = self.basis

        if basis is None:
            raise RuntimeError(
                'No basis defined or given. Use Morpher.set_basis(), Morpher.optimize_basis(), or the '
                'basis keyword.')

        n_benchmarks = len(basis)
        n_bases = n_benchmarks // self.n_components
        assert n_bases * self.n_components == n_benchmarks, 'Basis and number of components incompatible!'

        # Full morphing matrix. Will have shape (n_components, n_benchmarks) (note transposition later)
        morphing_matrix = np.zeros((n_benchmarks, self.n_components))

        # Morphing submatrix for each basis
        for i in range(n_bases):
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
            morphing_submatrix = morphing_submatrix / float(n_bases)

            # Write into full morphing matrix
            morphing_submatrix = morphing_submatrix.T
            morphing_matrix[i * n_benchmarks_this_basis:(i + 1) * n_benchmarks_this_basis] = morphing_submatrix

        morphing_matrix = morphing_matrix.T

        return morphing_matrix

    def calculate_morphing_weights(self, theta, basis=None, morphing_matrix=None):

        """Calculates the morphing weights w_b(theta) for a given basis {theta_b}

        Parameters
        ----------
        theta :
            
        basis :
             (Default value = None)
        morphing_matrix :
             (Default value = None)

        Returns
        -------

        """

        # Check all data is there
        if self.components is None or self.n_components is None or self.n_components <= 0:
            raise RuntimeError('No components defined. Use morpher.set_components() or morpher.find_components() '
                               'first!')

        if basis is None:
            basis = self.basis
            morphing_matrix = self.morphing_matrix

        if basis is None:
            raise RuntimeError(
                'No basis defined or given. Use Morpher.set_basis(), Morpher.optimize_basis(), or the '
                'basis keyword.')

        if morphing_matrix is None:
            morphing_matrix = self.calculate_morphing_matrix(basis)

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

    def calculate_morphing_weight_gradient(self, theta, basis=None, morphing_matrix=None):

        """Calculates the gradient of the morphing weights grad_i w_b(theta). Output shape is (gradient
        direction, basis benchmarks).

        Parameters
        ----------
        theta :
            
        basis :
             (Default value = None)
        morphing_matrix :
             (Default value = None)

        Returns
        -------

        """

        # Check all data is there
        if self.components is None or self.n_components is None or self.n_components <= 0:
            raise RuntimeError('No components defined. Use morpher.set_components() or morpher.find_components() '
                               'first!')

        if basis is None:
            basis = self.basis
            morphing_matrix = self.morphing_matrix

        if basis is None:
            raise RuntimeError(
                'No basis defined or given. Use Morpher.set_basis(), Morpher.optimize_basis(), or the '
                'basis keyword.')

        if morphing_matrix is None:
            morphing_matrix = self.calculate_morphing_matrix(basis)

        # Calculate gradients of component weights wrt theta
        component_weight_gradients = np.zeros((self.n_components, self.n_parameters))

        for c in range(self.n_components):
            for i in range(self.n_parameters):
                factor = 1.
                for p in range(self.n_parameters):
                    if p == i and self.components[c, p] > 0:
                        factor *= float(self.components[c, p]) * theta[p] ** (self.components[c, p] - 1)
                    elif p == i:
                        factor = 0.
                        break
                    else:
                        factor *= float(theta[p] ** self.components[c, p])
                component_weight_gradients[c, i] = factor

        # Transform to basis weights
        weight_gradients = morphing_matrix.T.dot(component_weight_gradients).T  # Shape (n_parameters, n_benchmarks)

        return weight_gradients

    def evaluate_morphing(self, basis=None, morphing_matrix=None, n_test_thetas=100,
                          return_weights_and_thetas=False):

        """Evaluates an objective function for a given basis

        Parameters
        ----------
        basis :
             (Default value = None)
        morphing_matrix :
             (Default value = None)
        n_test_thetas :
             (Default value = 100)
        return_weights_and_thetas :
             (Default value = False)

        Returns
        -------

        """

        # Check all data is there
        if self.components is None or self.n_components is None or self.n_components <= 0:
            raise RuntimeError('No components defined. Use morpher.set_components() or morpher.find_components() '
                               'first!')

        if basis is None:
            basis = self.basis
            morphing_matrix = self.morphing_matrix

        if basis is None:
            raise RuntimeError(
                'No basis defined or given. Use Morpher.set_basis(), Morpher.optimize_basis(), or the '
                'basis keyword.')

        if morphing_matrix is None:
            morphing_matrix = self.calculate_morphing_matrix(basis)

        thetas_test = self._draw_random_thetas(n_thetas=n_test_thetas)
        squared_weights = 0.
        squared_weight_list = []

        for theta in thetas_test:
            weights = self.calculate_morphing_weights(theta, basis, morphing_matrix)
            squared_weights += np.sum(weights * weights)

            if return_weights_and_thetas:
                squared_weight_list.append(np.sum(weights * weights))

        if return_weights_and_thetas:
            return thetas_test, np.array(squared_weight_list)

        squared_weights /= float(n_test_thetas)

        return -squared_weights
