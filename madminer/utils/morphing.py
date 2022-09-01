import itertools
import logging
import numpy as np

from collections import OrderedDict
from typing import Dict
from typing import Iterable

from madminer.models import Benchmark
from madminer.models import AnalysisParameter
from madminer.models import NuisanceParameter
from madminer.utils.various import sanitize_array

logger = logging.getLogger(__name__)


class PhysicsMorpher:
    """
    Morphing functionality for theory parameters. Morphing is a technique that allows MadMax to infer the full
    probability distribution `p(x_i | theta)` for each simulated event `x_i` and any `theta`, not just the benchmarks.

    For a typical MadMiner application, it is not necessary to use the morphing classes directly. The other MadMiner
    classes use the morphing functions "under the hood" when needed. Only for an isolated study of the morphing setup
    (e.g. to optimize the morphing basis), the PhysicsMorpher class itself may be of interest.

    A typical morphing basis setup involves the following steps:

    * The instance of the class is initialized with the parameter setup. The user can provide the parameters either
      in the format of `MadMiner.parameters`. Alternatively, human-friendly lists of the key properties can be provided.
    * The function `find_components` can be used to find the relevant components, i.e. individual terms
      contributing to the squared matrix elements (alternatively they can be defined by the user with
      `set_components()`).
    * The final step is the definition of the morphing basis, i.e. the benchmark points for which the squared matrix
      element will be evaluated before interpolating to other parameter points. Again the user can pick this basis
      manually with `set_basis()`. Alternatively, this class provides a basic optimization routine for the basis choice
      in `optimize_basis()`.

    The class also provides helper functions that are important for working with morphing:

    * `calculate_morphing_matrix()` calculates the morphing matrix, i.e. the matrix that links the morphing basis to the
       components.
    * `calculate_morphing_weights()` calculates the morphing weights `w_b(theta)` for a given parameter point `theta`
      such that `p(theta) = sum_b w_b(theta) p(theta_b)`.
    * `calculate_morphing_weight_gradient()` calculates the gradient of the morphing weights, `grad_theta w_b(theta)`.

    Note that this class only implements the "theory morphing" (or, more specifically, "EFT morphing") of the physics
    parameters of interest. Nuisance parameter morphing is implemented in the NuisanceMorpher class.

    Parameters
    ----------
    parameters_from_madminer : OrderedDict or None, optional
        Parameters in the `MadMiner.parameters` convention. OrderedDict with keys equal to the parameter names and
        values equal to AnalysisParameter model instances.

    parameter_max_power : None or list of int, optional
        Only used if parameters_from_madminer is None. Maximal power with which each parameter contributes to
        the squared matrix element. Typically at tree level, this maximal number is 2 for parameters
        that affect one vertex (e.g. only production or only decay of a particle),
        and 4 for parameters that affect two vertices (e.g. production and decay).

    parameter_range : None or list of tuple of float, optional
        Only used if parameters_from_madminer is None. Parameter range (param_min, param_max) for each parameter.
    """

    def __init__(
        self,
        parameters_from_madminer: Dict[str, AnalysisParameter] = None,
        parameter_max_power=None,
        parameter_range=None,
    ):

        # MadMiner interface
        if parameters_from_madminer is not None:
            self.use_madminer_interface = True
            self.n_parameters = len(parameters_from_madminer)
            self.parameter_names = [param.name for param in parameters_from_madminer.values()]
            self.parameter_max_power = [param.max_power for param in parameters_from_madminer.values()]
            self.parameter_range = [param.val_range for param in parameters_from_madminer.values()]

            self.parameter_max_power = np.array(self.parameter_max_power, dtype=int)
            self.parameter_range = np.array(self.parameter_range)

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

    def set_components(self, components):
        """
        Manually defines the components, i.e. the individual terms contributing to the squared matrix element.

        Parameters
        ----------
        components : ndarray
            Array with shape (n_components, n_parameters), where each entry gives the power with which a parameter
            scales a given component. For instance, a typical signal, interference, background situation with one
            parameter might be described by the components [[2], [1], [0]].

        Returns
        -------
            None
        """

        self.components = components
        self.n_components = len(self.components)

    def find_components(self, max_overall_power=4):
        """
        Finds the components, i.e. the individual terms contributing to the squared matrix element.

        Parameters
        ----------
        max_overall_power : int, optional
            The maximal sum of powers of all parameters contributing to the squared matrix element.
            Typically, if parameters can affect the couplings at n vertices, this number is 2n. Default value: 4.

        Returns
        -------
        components : ndarray
            Array with shape (n_components, n_parameters), where each entry gives the power with which a parameter
            scales a given component.
        """

        logger.debug("Max overall power %s", max_overall_power)
        logger.debug("Max individual power %s", [max_power for max_power in self.parameter_max_power])

        powers_each_component = [range(max_power + 1) for max_power in self.parameter_max_power]

        # Go through regions and finds components for each
        components = []
        for powers in itertools.product(*powers_each_component):
            powers = np.array(powers, dtype=int)

            if np.sum(powers) > max_overall_power:
                continue

            if not any((powers == x).all() for x in components):
                logger.debug("  Adding component %s", powers)
                components.append(np.copy(powers))
            else:
                logger.debug("  Not adding component %s again", powers)

        self.components = np.array(components, dtype=int)
        self.n_components = len(self.components)

        return self.components

    def set_basis(self, basis_from_madminer: Dict[str, Benchmark] = None, basis_numpy=None, morphing_matrix=None):
        """
        Manually sets the basis benchmarks.

        Parameters
        ----------
        basis_from_madminer : OrderedDict or None, optional
            Basis in the `MadMiner.benchmarks` conventions. Default value: None.

        basis_numpy : ndarray or None, optional
            Only used if basis_from_madminer is None. Basis as a ndarray with shape (n_components, n_parameters).

        morphing_matrix : ndarray or None, optional
            Manually provided morphing matrix. If None, the morphing matrix is calculated automatically. Default value:
            None.

        Returns
        -------
            None
        """

        if basis_from_madminer is not None:
            self.basis = []
            for benchmark in basis_from_madminer.values():
                self.basis.append([benchmark.values[key] for key in self.parameter_names])
            self.basis = np.array(self.basis)
        elif basis_numpy is not None:
            self.basis = np.array(basis_numpy)
        else:
            raise RuntimeError("No basis given")

        # Restrict basis to the first benchmarks
        self.basis = self.basis[: self.n_components, :]

        if morphing_matrix is None:
            self.morphing_matrix = self.calculate_morphing_matrix()
        else:
            self.morphing_matrix = morphing_matrix

    def optimize_basis(
        self,
        n_bases=1,
        benchmarks_from_madminer: Dict[str, Benchmark] = None,
        benchmarks_numpy=None,
        n_trials=100,
        n_test_thetas=100,
    ):
        """
        Optimizes the morphing basis. If either fixed_benchmarks_from_madminer or fixed_benchmarks_numpy are not
        None, then these will be used as fixed basis points and only the remaining part of the basis will be optimized.

        Parameters
        ----------
        n_bases : int, optional
            The number of morphing bases generated. If n_bases > 1, multiple bases are combined, and the
            weights for each basis are reduced by a factor 1 / n_bases. Currently only the default choice of 1 is
            fully implemented. Do not use any other value for now. Default value: 1.

        benchmarks_from_madminer : OrderedDict or None, optional
            Input basis vectors in the `MadMiner.benchmarks` conventions. Default value: None.

        benchmarks_numpy : ndarray or None, optional
            Input basis vectors as a ndarray with shape `(n_fixed_basis_points, n_parameters)`. Default value: None.

        n_trials : int, optional
            Number of random basis configurations tested in the optimization procedure. A larger number will increase
            the run time of the optimization, but lead to better results. Default value: 100.

        n_test_thetas : int, optional
            Number of random parameter points used to evaluate the expected mean squared morphing weights. A larger
            number will increase the run time of the optimization, but lead to better results. Default value: 100.

        Returns
        -------
        basis : OrderedDict or ndarray
            Optimized basis in the same format (MadMiner or numpy) as the parameters provided during instantiation.
        """

        # Check all data is there
        if self.components is None or self.n_components is None or self.n_components <= 0:
            raise RuntimeError(
                "No components defined. Use morpher.set_components() or morpher.find_components() " "first!"
            )

        # Fixed benchmarks
        if benchmarks_from_madminer is not None:
            fixed_benchmarks = []
            fixed_benchmark_names = []
            for benchmark in benchmarks_from_madminer.values():
                fixed_benchmark_names.append(benchmark.name)
                fixed_benchmarks.append([benchmark.values[key] for key in self.parameter_names])
            fixed_benchmarks = np.array(fixed_benchmarks)
        elif benchmarks_numpy is not None:
            fixed_benchmarks = np.array(benchmarks_numpy)
            fixed_benchmark_names = []
        else:
            fixed_benchmarks = np.array([])
            fixed_benchmark_names = []

        # Missing benchmarks
        n_benchmarks = n_bases * self.n_components
        n_missing_benchmarks = n_benchmarks - len(fixed_benchmarks)

        assert n_missing_benchmarks >= 0, "Too many fixed benchmarks!"

        # Optimization
        best_basis = None
        best_morphing_matrix = None
        best_performance = None

        for i in range(n_trials):
            basis = self._propose_basis(fixed_benchmarks, n_missing_benchmarks)
            morphing_matrix = self.calculate_morphing_matrix(basis)
            performance = self.evaluate_morphing(basis, morphing_matrix, n_test_thetas=n_test_thetas)

            if (
                best_performance is None
                or best_basis is None
                or best_morphing_matrix is None
                or performance > best_performance
            ):
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
                    benchmark_name = f"morphing_basis_vector_{len(basis_madminer)}"
                parameter = OrderedDict()
                for p, p_name in enumerate(self.parameter_names):
                    parameter[p_name] = benchmark[p]
                basis_madminer[benchmark_name] = parameter

            return basis_madminer

        # Normal output
        return best_basis

    def calculate_morphing_matrix(self, basis=None):
        """
        Calculates the morphing matrix that links the components to the basis benchmarks.

        Parameters
        ----------
        basis : ndarray or None, optional
             Manually specified morphing basis for which the morphing matrix is calculated. This array has shape
             `(n_basis_benchmarks, n_parameters)`. If None, the basis from the last call of `set_basis()` or
             `find_basis()` is used. Default value: None.

        Returns
        -------
        morphing_matrix : ndarray
            Morphing matrix with shape `(n_basis_benchmarks, n_components)`
        """

        # Check all data is there
        if self.components is None or self.n_components is None or self.n_components <= 0:
            raise RuntimeError(
                "No components defined. Use morpher.set_components() or morpher.find_components() " "first!"
            )

        if basis is None:
            basis = self.basis

        if basis is None:
            raise RuntimeError(
                "No basis defined or given. Use PhysicsMorpher.set_basis(), PhysicsMorpher.optimize_basis(), or the "
                "basis keyword."
            )

        n_benchmarks = len(basis)
        n_bases = n_benchmarks // self.n_components
        assert n_bases * self.n_components == n_benchmarks, "Basis and number of components incompatible!"

        # Full morphing matrix. Will have shape (n_components, n_benchmarks_phys) (note transposition later)
        morphing_matrix = np.zeros((n_benchmarks, self.n_components))

        # Morphing submatrix for each basis
        for i in range(n_bases):
            n_benchmarks_this_basis = self.n_components
            this_basis = basis[i * n_benchmarks_this_basis : (i + 1) * n_benchmarks_this_basis]

            inv_morphing_submatrix = np.zeros((n_benchmarks_this_basis, self.n_components))

            for b in range(n_benchmarks_this_basis):
                for c in range(self.n_components):
                    factor = 1.0
                    for p in range(self.n_parameters):
                        factor *= float(this_basis[b, p] ** self.components[c, p])
                    inv_morphing_submatrix[b, c] = factor

            # Invert -? components expressed in basis points. Shape (n_components, n_benchmarks_this_basis)
            morphing_submatrix = np.linalg.inv(inv_morphing_submatrix)

            # For now, just use 1 / n_bases times the weights of each basis
            morphing_submatrix = morphing_submatrix / float(n_bases)

            # Write into full morphing matrix
            morphing_submatrix = morphing_submatrix.T
            morphing_matrix[i * n_benchmarks_this_basis : (i + 1) * n_benchmarks_this_basis] = morphing_submatrix

        return morphing_matrix.T

    def calculate_morphing_weights(self, theta, basis=None, morphing_matrix=None):
        """
        Calculates the morphing weights `w_b(theta)` for a given morphing basis `{theta_b}`.

        Parameters
        ----------
        theta : ndarray
            Parameter point `theta` with shape `(n_parameters,)`.

        basis : ndarray or None, optional
             Manually specified morphing basis for which the weights are calculated. This array has shape
             `(n_basis_benchmarks, n_parameters)`. If None, the basis from the last call of `set_basis()` or
             `find_basis()` is used. Default value: None.

        morphing_matrix : ndarray or None, optional
             Manually specified morphing matrix for the given morphing basis. This array has shape
             `(n_basis_benchmarks, n_components)`. If None, the morphing matrix is calculated automatically. Default
             value: None.

        Returns
        -------
        morphing_weights : ndarray
            Morphing weights as an array with shape `(n_basis_benchmarks,)`.
        """

        # Check all data is there
        if self.components is None or self.n_components is None or self.n_components <= 0:
            raise RuntimeError(
                "No components defined. Use morpher.set_components() or morpher.find_components() " "first!"
            )

        if basis is None:
            basis = self.basis
            morphing_matrix = self.morphing_matrix

        if basis is None:
            raise RuntimeError(
                "No basis defined or given. Use PhysicsMorpher.set_basis(), PhysicsMorpher.optimize_basis(), or the "
                "basis keyword."
            )

        if morphing_matrix is None:
            morphing_matrix = self.calculate_morphing_matrix(basis)

        # Calculate component weights
        component_weights = np.zeros(self.n_components)
        for c in range(self.n_components):
            factor = 1.0
            for p in range(self.n_parameters):
                factor *= float(theta[p] ** self.components[c, p])
            component_weights[c] = factor
        component_weights = np.array(component_weights)

        # Transform to basis weights
        return morphing_matrix.T.dot(component_weights)

    def calculate_morphing_weight_gradient(self, theta, basis=None, morphing_matrix=None):
        """
        Calculates the gradient of the morphing weights, `grad_i w_b(theta)`.

        Parameters
        ----------
        theta : ndarray
            Parameter point `theta` with shape `(n_parameters,)`.

        basis : ndarray or None, optional
             Manually specified morphing basis for which the weights are calculated. This array has shape
             `(n_basis_benchmarks, n_parameters)`. If None, the basis from the last call of `set_basis()` or
             `find_basis()` is used. Default value: None.

        morphing_matrix : ndarray or None, optional
             Manually specified morphing matrix for the given morphing basis. This array has shape
             `(n_basis_benchmarks, n_components)`. If None, the morphing matrix is calculated automatically. Default
             value: None.

        Returns
        -------
        morphing_weight_gradients : ndarray
            Morphing weights as an array with shape `(n_parameters, n_basis_benchmarks,)`, where the first component
            refers to the gradient direction.
        """

        # Check all data is there
        if self.components is None or self.n_components is None or self.n_components <= 0:
            raise RuntimeError(
                "No components defined. Use morpher.set_components() or morpher.find_components() " "first!"
            )

        if basis is None:
            basis = self.basis
            morphing_matrix = self.morphing_matrix

        if basis is None:
            raise RuntimeError(
                "No basis defined or given. Use PhysicsMorpher.set_basis(), PhysicsMorpher.optimize_basis(), or the "
                "basis keyword."
            )

        if morphing_matrix is None:
            morphing_matrix = self.calculate_morphing_matrix(basis)

        # Calculate gradients of component weights wrt theta
        component_weight_gradients = np.zeros((self.n_components, self.n_parameters))

        for c in range(self.n_components):
            for i in range(self.n_parameters):
                factor = 1.0
                for p in range(self.n_parameters):
                    if p == i and self.components[c, p] > 0:
                        factor *= float(self.components[c, p]) * theta[p] ** (self.components[c, p] - 1)
                    elif p == i:
                        factor = 0.0
                        break
                    else:
                        factor *= float(theta[p] ** self.components[c, p])
                component_weight_gradients[c, i] = factor

        # Transform to basis weights
        # Shape (n_parameters, n_benchmarks_phys)
        return morphing_matrix.T.dot(component_weight_gradients).T

    def evaluate_morphing(self, basis=None, morphing_matrix=None, n_test_thetas=100, return_weights_and_thetas=False):
        """
        Evaluates the expected sum of the squared morphing weights for a given basis.

        Parameters
        ----------
        basis : ndarray or None, optional
             Manually specified morphing basis for which the weights are calculated. This array has shape
             `(n_basis_benchmarks, n_parameters)`. If None, the basis from the last call of `set_basis()` or
             `find_basis()` is used. Default value: None.

        morphing_matrix : ndarray or None, optional
             Manually specified morphing matrix for the given morphing basis. This array has shape
             `(n_basis_benchmarks, n_components)`. If None, the morphing matrix is calculated automatically. Default
             value: None.

        n_test_thetas : int, optional
            Number of random parameter points used to evaluate the expected mean squared morphing weights. A larger
            number will increase the run time of the optimization, but lead to better results. Default value: 100.

        return_weights_and_thetas : bool, optional
             If True, results for each evaluation theta are returned, rather than taking their average. Default value:
             False.

        Returns
        -------
        thetas_test : ndarray
            Random parameter points used for evaluation. Only returned if `return_weights_and_thetas=True` is used.

        squared_weights : ndarray
            Squared summed morphing weights at each evaluation parameter point. Only returned if
            `return_weights_and_thetas=True` is used.

        negative_expected_sum_squared_weights : float
            Negative expected sum of the square of the morphing weights. Objective function in the optimization.
            Only returned with `return_weights_and_thetas=False`.
        """

        # Check all data is there
        if self.components is None or self.n_components is None or self.n_components <= 0:
            raise RuntimeError(
                "No components defined. Use morpher.set_components() or morpher.find_components() " "first!"
            )

        if basis is None:
            basis = self.basis
            morphing_matrix = self.morphing_matrix

        if basis is None:
            raise RuntimeError(
                "No basis defined or given. Use PhysicsMorpher.set_basis(), PhysicsMorpher.optimize_basis(), or the "
                "basis keyword."
            )

        if morphing_matrix is None:
            morphing_matrix = self.calculate_morphing_matrix(basis)

        thetas_test = self._draw_random_thetas(n_thetas=n_test_thetas)
        squared_weights = 0.0
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

    def _propose_basis(self, fixed_benchmarks, n_missing_benchmarks):
        """ Proposes a random basis. """

        if len(fixed_benchmarks) > 0:
            basis = np.vstack([fixed_benchmarks, self._draw_random_thetas(n_missing_benchmarks)])
        else:
            basis = self._draw_random_thetas(n_missing_benchmarks)

        return basis

    def _draw_random_thetas(self, n_thetas):
        """ Randomly draws parameter vectors within the specified parameter ranges. """

        # First draw random numbers in range [0, 1)^n_parameters
        u = np.random.rand(n_thetas, self.n_parameters)

        # Transform to right range
        thetas = self.parameter_range[:, 0] + u * (self.parameter_range[:, 1] - self.parameter_range[:, 0])

        return thetas


class NuisanceMorpher:
    """
    Morphing functionality for nuisance parameters.

    For a typical MadMiner application, it is not necessary to use the morphing classes directly. The other MadMiner
    classes use the morphing functions "under the hood" when needed.

    Parameters
    ----------
    nuisance_parameters_from_madminer : OrderedDict
        Nuisance parameters defined in the form {name: (benchmark_name_pos, benchmark_name_neg)}. Here
        benchmark_name_pos refers to the name of the benchmark with nu_i = 1, while benchmark_name_neg is either None
        or refers to the name of the benchmark with nu_i = -1.

    benchmark_names : list
        The names of the benchmarks.

    reference_benchmark : str
        Name of the reference benchmark.
    """

    def __init__(
        self,
        nuisance_parameters_from_madminer: Dict[str, NuisanceParameter],
        benchmark_names: Iterable[str],
        reference_benchmark: str
    ):

        # Benchmarks
        self.benchmark_names = list(benchmark_names)
        self.i_benchmark_ref = list(benchmark_names).index(reference_benchmark)

        # Nuisance parameters
        self.nuisance_parameters = nuisance_parameters_from_madminer
        self.n_nuisance_parameters = len(self.nuisance_parameters)

        self.i_benchmarks_pos = []
        self.i_benchmarks_neg = []
        self.degrees = []

        for param in self.nuisance_parameters.values():
            degrees = 0

            if param.benchmark_pos is not None:
                degrees += 1
                index_benchmark_pos = self.benchmark_names.index(param.benchmark_pos)
            else:
                index_benchmark_pos = None

            if param.benchmark_neg is not None:
                degrees += 1
                index_benchmark_neg = self.benchmark_names.index(param.benchmark_neg)
            else:
                index_benchmark_neg = None

            self.i_benchmarks_pos.append(index_benchmark_pos)
            self.i_benchmarks_neg.append(index_benchmark_neg)
            self.degrees.append(degrees)

    def calculate_a(self, benchmark_weights):
        """
        Calculates the first-order coefficients a_i(x) in
        `dsigma(x |  theta, nu) / dsigma(x | theta, 0) = exp[ sum_i (a_i(x) nu_i + b_i(x) nu_i^2 )]`.

        Parameters
        ----------
        benchmark_weights : ndarray
            Event weights `dsigma(x | theta_i, nu_i)` with shape `(n_events, n_benchmarks)`. The benchmarks are expected
            to be sorted in the same order as the keyword benchmark_names used during initialization, and the
            nuisance benchmarks are expected to be rescaled to have the same physics parameters theta as the
            reference_benchmark given during initialization.

        Returns
        -------
        a : ndarray
            Coefficients a_i(x) with shape `(n_nuisance_parameters, n_events)`.
        """

        a = []

        with np.errstate(divide="ignore", invalid="ignore"):
            for i_pos, i_neg, degree in zip(self.i_benchmarks_pos, self.i_benchmarks_neg, self.degrees):
                if degree == 1:
                    a.append(np.log(benchmark_weights[:, i_pos] / benchmark_weights[:, self.i_benchmark_ref]))
                elif degree == 2:
                    a.append(0.5 * np.log(benchmark_weights[:, i_pos] / benchmark_weights[:, i_neg]))

        a = np.array(a)  # Shape (n_nuisance_parameters, n_events)
        a = sanitize_array(a, min_value=-10.0, max_value=10.0)
        return a

    def calculate_b(self, benchmark_weights):
        """
        Calculates the second-order coefficients b_i(x) in
        `dsigma(x |  theta, nu) / dsigma(x | theta, 0) = exp[ sum_i (a_i(x) nu_i + b_i(x) nu_i^2 )]`.

        Parameters
        ----------
        benchmark_weights : ndarray
            Event weights `dsigma(x | theta_i, nu_i)` with shape `(n_events, n_benchmarks)`. The benchmarks are expected
            to be sorted in the same order as the keyword benchmark_names used during initialization, and the
            nuisance benchmarks are expected to be rescaled to have the same physics parameters theta as the
            reference_benchmark given during initialization.

        Returns
        -------
        b : ndarray
            Coefficients b_i(x) with shape `(n_nuisance_parameters, n_events)`.
        """

        b = []

        for i_pos, i_neg, degree in zip(self.i_benchmarks_pos, self.i_benchmarks_neg, self.degrees):
            if degree == 1:
                b.append(np.zeros(benchmark_weights.shape[0]))
            elif degree == 2:
                b.append(
                    0.5
                    * np.log(
                        benchmark_weights[:, i_pos]
                        * benchmark_weights[:, i_neg]
                        / benchmark_weights[:, self.i_benchmark_ref] ** 2
                    )
                )

        b = np.array(b)  # Shape (n_nuisance_parameters, n_events)
        b = sanitize_array(b, min_value=-10.0, max_value=10.0)
        return b

    def calculate_nuisance_factors(self, nuisance_parameters, benchmark_weights):
        """
        Calculates the rescaling of the event weights from non-central values of nuisance parameters.

        Parameters
        ----------
        nuisance_parameters : ndarray
            Values of the nuisance parameters `nu`, with shape `(n_nuisance_parameters,)`.

        benchmark_weights : ndarray
            Event weights `dsigma(x | theta_i, nu_i)` with shape `(n_events, n_benchmarks)`. The benchmarks are expected
            to be sorted in the same order as the keyword benchmark_names used during initialization, and the
            nuisance benchmarks are expected to be rescaled to have the same physics parameters theta as the
            reference_benchmark given during initialization.

        Returns
        -------
        nuisance_factors : ndarray
            Nuisance factor `dsigma(x |  theta, nu) / dsigma(x | theta, 0)` with shape `(n_events,)`.
        """

        if nuisance_parameters is None:
            nuisance_parameters = np.zeros(self.n_nuisance_parameters)

        a = self.calculate_a(benchmark_weights)  # Shape (n_nuisance_parameters, n_events)
        b = self.calculate_b(benchmark_weights)  # Shape (n_nuisance_parameters, n_events)

        exponent = np.sum(a * nuisance_parameters[:, np.newaxis] + b * nuisance_parameters[:, np.newaxis] ** 2, axis=0)
        nuisance_factors = np.exp(exponent)

        return nuisance_factors

    def calculate_log_nuisance_factor_gradients(self, nuisance_parameters, benchmark_weights):
        """
        Calculates the gradient of the log of the nuisance factors with respect to the nuisance parameters.

        Parameters
        ----------
        nuisance_parameters : ndarray
            Values of the nuisance parameters `nu`, with shape `(n_nuisance_parameters,)`.

        benchmark_weights : ndarray
            Event weights `dsigma(x | theta_i, nu_i)` with shape `(n_events, n_benchmarks)`. The benchmarks are expected
            to be sorted in the same order as the keyword benchmark_names used during initialization, and the
            nuisance benchmarks are expected to be rescaled to have the same physics parameters theta as the
            reference_benchmark given during initialization.

        Returns
        -------
        log_nuisance_factor_gradients : ndarray
            Log nuisance factor gradients `grad_nu log (dsigma(x | theta, nu) / dsigma(x | theta, 0))` with shape
            `(n_parameters, n_events)`.
        """

        if nuisance_parameters is None:
            nuisance_parameters = np.zeros(self.n_nuisance_parameters)

        a = self.calculate_a(benchmark_weights)  # Shape (n_nuisance_parameters, n_events)
        b = self.calculate_b(benchmark_weights)  # Shape (n_nuisance_parameters, n_events)

        log_gradients = a + 2.0 * b * nuisance_parameters[:, np.newaxis]

        return log_gradients

    def calculate_nuisance_factor_gradients(self, nuisance_parameters, benchmark_weights):
        """
        Calculates the gradient of the nuisance factors with respect to the nuisance parameters.

        Parameters
        ----------
        nuisance_parameters : ndarray
            Values of the nuisance parameters `nu`, with shape `(n_nuisance_parameters,)`.

        benchmark_weights : ndarray
            Event weights `dsigma(x | theta_i, nu_i)` with shape `(n_events, n_benchmarks)`. The benchmarks are expected
            to be sorted in the same order as the keyword benchmark_names used during initialization, and the
            nuisance benchmarks are expected to be rescaled to have the same physics parameters theta as the
            reference_benchmark given during initialization.

        Returns
        -------
        nuisance_factor_gradients : ndarray
            Nuisance factor gradients `grad_nu (dsigma(x | theta, nu) / dsigma(x | theta, 0))` with shape
            `(n_parameters, n_events)`.
        """

        if nuisance_parameters is None:
            nuisance_parameters = np.zeros(self.n_nuisance_parameters)

        a = self.calculate_a(benchmark_weights)  # Shape (n_nuisance_parameters, n_events)
        b = self.calculate_b(benchmark_weights)  # Shape (n_nuisance_parameters, n_events)

        exponent = np.sum(a * nuisance_parameters[:, np.newaxis] + b * nuisance_parameters[:, np.newaxis] ** 2, axis=0)
        nuisance_factors = np.exp(exponent)
        log_gradients = a + 2.0 * b * nuisance_parameters[:, np.newaxis]
        gradients = log_gradients * nuisance_factors[np.newaxis, :]

        return gradients
