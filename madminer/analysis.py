from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
import collections
import six

from madminer.utils.interfaces.madminer_hdf5 import load_madminer_settings, madminer_event_loader
from madminer.utils.analysis import get_theta_benchmark_matrix, get_dtheta_benchmark_matrix
from madminer.utils.analysis import mdot
from madminer.utils.morphing import PhysicsMorpher, NuisanceMorpher
from madminer.utils.various import format_benchmark

logger = logging.getLogger(__name__)


class EventAnalyzer(object):
    """
    Collects common functionality that is used when analysing data in the MadMiner file.

    Parameters
    ----------
    filename : str
        Path to MadMiner file (for instance the output of `madminer.delphes.DelphesProcessor.save()`).

    disable_morphing : bool, optional
        If True, the morphing setup is not loaded from the file. Default value: False.

    include_nuisance_parameters : bool, optional
        If True, nuisance parameters are taken into account. Default value: True.

    """

    def __init__(self, filename, disable_morphing=False, include_nuisance_parameters=True):
        # Save setup
        self.include_nuisance_parameters = include_nuisance_parameters
        self.madminer_filename = filename

        logger.info("Loading data from %s", filename)

        # Load data
        (
            self.parameters,
            self.benchmarks,
            self.benchmark_is_nuisance,
            self.morphing_components,
            self.morphing_matrix,
            self.observables,
            self.n_samples,
            _,
            self.reference_benchmark,
            self.nuisance_parameters,
        ) = load_madminer_settings(filename, include_nuisance_benchmarks=include_nuisance_parameters)

        self.n_parameters = len(self.parameters)
        self.n_benchmarks = len(self.benchmarks)
        self.n_benchmarks_phys = np.sum(np.logical_not(self.benchmark_is_nuisance))
        self.n_observables = len(self.observables)

        self.n_nuisance_parameters = 0
        if self.nuisance_parameters is not None and include_nuisance_parameters:
            self.n_nuisance_parameters = len(self.nuisance_parameters)
        else:
            self.nuisance_parameters = None

        logger.info("Found %s parameters", self.n_parameters)
        for key, values in six.iteritems(self.parameters):
            logger.debug(
                "   %s (LHA: %s %s, maximal power in squared ME: %s, range: %s)",
                key,
                values[0],
                values[1],
                values[2],
                values[3],
            )

        if self.nuisance_parameters is not None:
            logger.info("Found %s nuisance parameters", self.n_nuisance_parameters)
            for key, values in six.iteritems(self.nuisance_parameters):
                logger.debug("   %s (%s)", key, values)
        else:
            logger.info("Did not find nuisance parameters")

        logger.info("Found %s benchmarks, of which %s physical", self.n_benchmarks, self.n_benchmarks_phys)
        for (key, values), is_nuisance in zip(six.iteritems(self.benchmarks), self.benchmark_is_nuisance):
            if is_nuisance:
                logger.debug("   %s: nuisance parameter", key)
            else:
                logger.debug("   %s: %s", key, format_benchmark(values))

        logger.info("Found %s observables", len(self.observables))
        for i, obs in enumerate(self.observables):
            logger.debug("  %2.2s %s", i, obs)
        logger.info("Found %s events", self.n_samples)

        # Morphing
        self.morpher = None
        if self.morphing_matrix is not None and self.morphing_components is not None and not disable_morphing:
            self.morpher = PhysicsMorpher(self.parameters)
            self.morpher.set_components(self.morphing_components)
            self.morpher.set_basis(self.benchmarks, morphing_matrix=self.morphing_matrix)

            logger.info("Found morphing setup with %s components", len(self.morphing_components))

        else:
            logger.info("Did not find morphing setup.")

        # Nuisance morphing
        self.nuisance_morpher = None
        if self.nuisance_parameters is not None:
            self.nuisance_morpher = NuisanceMorpher(
                self.nuisance_parameters, list(self.benchmarks.keys()), self.reference_benchmark
            )
            logger.info("Found nuisance morphing setup")

    def weighted_events(self, theta=None, nu=None, start_event=None, end_event=None, derivative=False):
        """
        Returns all events together with the benchmark weights (if theta is None) or weights for a given theta.

        Parameters
        ----------
        theta : None or ndarray or str, optional
            If None, the function returns all benchmark weights. If str, the function returns the weights for a given
            benchmark name. If ndarray, it uses morphing to calculate the weights for this value of theta. Default
            value: None.

        nu : None or ndarray, optional
            If None, the nuisance parameters are set to their nominal values. Otherwise, and if theta is an ndarray,
            sets the values of the nuisance parameters.

        start_event : int
            Index (in the MadMiner file) of the first event to consider.

        end_event : int
            Index (in the MadMiner file) of the last unweighted event to consider.

        derivative : bool, optional
            If True and if theta is not None, the derivative of the weights with respect to theta are returned. Default
            value: False.

        Returns
        -------
        x : ndarray
            Observables with shape `(n_unweighted_samples, n_observables)`.

        weights : ndarray
            If theta is None and derivative is False, benchmark weights with shape
            `(n_unweighted_samples, n_benchmarks)` in pb. If theta is not None and derivative is True, the gradient of
            the weight for the given parameter with respect to theta with shape `(n_unweighted_samples, n_gradients)`
            in pb. Otherwise, weights for the given parameter theta with shape `(n_unweighted_samples,)` in pb.

        """

        x, weights_benchmarks = next(
            madminer_event_loader(self.madminer_filename, batch_size=None, start=start_event, end=end_event)
        )

        if theta is None:
            return x, weights_benchmarks

        elif isinstance(theta, six.string_types):
            i_benchmark = list(self.benchmarks.keys()).index(theta)
            return x, weights_benchmarks[:, i_benchmark]

        elif derivative:
            dtheta_matrix = get_dtheta_benchmark_matrix("morphing", theta, self.benchmarks, self.morpher)

            gradients_theta = mdot(dtheta_matrix, weights_benchmarks)  # (n_gradients, n_samples)
            gradients_theta = gradients_theta.T

            return x, gradients_theta

        else:
            # TODO: nuisance params
            if nu is not None:
                raise NotImplementedError

            theta_matrix = get_theta_benchmark_matrix("morphing", theta, self.benchmarks, self.morpher)

            weights_theta = mdot(theta_matrix, weights_benchmarks)

            return x, weights_theta

    def xsecs(
        self,
        thetas=None,
        nus=None,
        start_event=None,
        end_event=None,
        include_nuisance_benchmarks=False,
        batch_size=100000,
    ):
        """
        Returns the total cross sections for benchmarks or parameter points.

        Parameters
        ----------
        thetas : None or list of (ndarray or str), optional
            If None, the function returns all benchmark cross sections. Otherwise, it returns the cross sections for a
            series of parameter points that are either given by their benchmark name (as a str), their benchmark index
            (as an int), or their parameter value (as an ndarray, using morphing). Default value: None.

        nus : None or list of (None or ndarray), optional
             If None, the nuisance parameters are set to their nominal values (0), i.e. no systematics are taken into
             account. Otherwise, the list has to have the same number of elements as thetas, and each entry can specify
             nuisance parameters at nominal value (None) or a value of the nuisance parameters (ndarray).

        start_event : int or None, optional
            Index (in the MadMiner file) of the first event to consider. Default value: None.

        end_event : int or None, optional
            Index (in the MadMiner file) of the last unweighted event to consider. Default value: None.

        include_nuisance_benchmarks : bool, optional
            Whether to includee nuisance benchmarks if thetas is None. Default value: False.

        batch_size : int, optional
            Size of the batches of events that are loaded into memory at the same time. Default value: 100000.

        Returns
        -------
        xsecs : ndarray
            Calculated cross sections in pb.

        xsec_uncertainties : ndarray
            Cross-section uncertainties in pb. Basically calculated as sum(weights**2)**0.5.
        """

        logger.debug("Calculating cross sections for thetas = %s and nus = %s", thetas, nus)

        # Inputs
        if thetas is not None:
            include_nuisance_benchmarks = nus is not None

        if thetas is not None:
            if nus is None:
                nus = [None for _ in thetas]
            assert len(nus) == len(thetas), "Numbers of thetas and nus don't match!"

        # Theta matrices (translation of benchmarks to theta, at nominal nuisance params)
        theta_matrices = []
        for theta in thetas:
            if isinstance(theta, six.string_types):
                i_benchmark = list(self.benchmarks.keys()).index(theta)
                theta_matrix = get_theta_benchmark_matrix("benchmark", i_benchmark, self.benchmarks)
            elif isinstance(theta, int):
                theta_matrix = get_theta_benchmark_matrix("benchmark", theta, self.benchmarks)
            else:
                theta_matrix = get_theta_benchmark_matrix("morphing", theta, self.benchmarks, self.morpher)
            theta_matrices.append(theta_matrix)
        theta_matrices = np.asarray(theta_matrices)  # Shape (n_thetas, n_benchmarks)

        # Loop over events
        xsecs = 0.0
        xsec_uncertainties = 0.0

        for i_batch, (_, benchmark_weights) in enumerate(
            madminer_event_loader(
                self.madminer_filename,
                start=start_event,
                end=end_event,
                include_nuisance_parameters=include_nuisance_benchmarks,
                benchmark_is_nuisance=self.benchmark_is_nuisance,
                batch_size=batch_size,
            )
        ):
            n_batch, _ = benchmark_weights.shape
            logger.debug("Batch %s with %s events", i_batch + 1, n_batch)

            # Benchmark xsecs
            if thetas is None:
                xsecs += np.sum(benchmark_weights, axis=0)
                xsec_uncertainties += np.sum(benchmark_weights * benchmark_weights, axis=0)

            # xsecs at given parame ters(theta, nu)
            else:
                # Weights at nominal nuisance params (nu=0)
                weights_nom = mdot(theta_matrices, benchmark_weights)  # Shape (n_thetas, n_batch)
                weights_sq_nom = mdot(theta_matrices, benchmark_weights * benchmark_weights)  # same

                logger.debug("Nominal weights: %s", weights_nom)

                # Effect of nuisance parameters
                nuisance_factors = []
                for nu in nus:
                    if nu is None:
                        nuisance_factors.append(np.ones(n_batch))
                    else:
                        nuisance_factors.append(self.nuisance_morpher.calculate_nuisance_factors(nu, benchmark_weights))
                nuisance_factors = np.asarray(nuisance_factors)  # Shape (n_thetas, n_batch)

                logger.debug("Nuisance factors: %s", nuisance_factors)

                weights = nuisance_factors * weights_nom
                weights_sq = nuisance_factors * weights_sq_nom

                # Sum up
                xsecs += np.sum(weights, axis=1)
                xsec_uncertainties += np.sum(weights_sq, axis=1)

            xsec_uncertainties = xsec_uncertainties ** 0.5

            logger.debug("xsecs and uncertainties [pb]:")
            for this_xsec, this_uncertainty in zip(xsecs, xsec_uncertainties):
                logger.debug("  %s +/- %s (%s %)", this_xsec, this_uncertainty, 100 * this_uncertainty / this_xsec)

            return xsecs, xsec_uncertainties

    def _calculate_benchmark_xsecs_sampling(self, start_event, end_event, use_nuisance_parameters):
        xsecs_benchmarks = 0.0
        squared_weight_sum_benchmarks = 0.0
        n_observables = 0
        for obs, weights in madminer_event_loader(
            self.madminer_filename,
            start=start_event,
            end=end_event,
            include_nuisance_parameters=use_nuisance_parameters,
            benchmark_is_nuisance=self.benchmark_is_nuisance,
        ):
            xsecs_benchmarks += np.sum(weights, axis=0)
            squared_weight_sum_benchmarks += np.sum(weights * weights, axis=0)
            n_observables = obs.shape[1]
        return xsecs_benchmarks, squared_weight_sum_benchmarks, n_observables

    def _calculate_xsec_fisherinformation(
        self,
        theta=None,
        cuts=None,
        efficiency_functions=None,
        return_benchmark_xsecs=False,
        return_error=False,
        include_nuisance_parameters=True,
        start_event=0,
    ):
        """
        Calculates the total cross section for a parameter point.

        Parameters
        ----------
        theta : ndarray or None, optional
            The parameter point. If None, return_benchmark_xsecs should be True. Default value: None.

        cuts : list of str or None, optional
            Cuts. Each entry is a parseable Python expression that returns a bool (True if the event should pass a cut,
            False otherwise). Default value: None.

        efficiency_functions : list of str or None
            Efficiencies. Each entry is a parseable Python expression that returns a float for the efficiency of one
            component. Default value: None.

        return_benchmark_xsecs : bool, optional
            If True, this function returns the benchmark xsecs. Otherwise, it returns the xsec at theta. Default value:
            False.

        return_error : bool, optional
            If True, this function also returns the square root of the summed squared weights.

        include_nuisance_parameters : bool, optional
            If True and if return_benchmark_xsecs is True, the nuisance benchmarks are included in the output. Default
            value: True.

        start_event : int, optional
            Index of first event in MadMiner file to consider. Default value: 0.

        Returns
        -------
        xsec : ndarray or float
            If return_benchmark_xsecs is True, an ndarray of benchmark xsecs in pb is returned. Otherwise, the cross
            section at theta in pb is returned.

        xsec_uncertainty : ndarray or float
            Only returned if return_error is True. Uncertainty (square root of the summed squared weights) on xsec.

        """

        logger.debug("Calculating total cross section for theta = %s", theta)

        # Input
        if cuts is None:
            cuts = []
        if efficiency_functions is None:
            efficiency_functions = []

        assert (theta is not None) or return_benchmark_xsecs, "Please supply theta or set return_benchmark_xsecs=True"

        # Total xsecs for benchmarks
        xsecs_benchmarks = None
        xsecs_uncertainty_benchmarks = None

        for observations, weights in madminer_event_loader(
            self.madminer_filename, start=start_event, include_nuisance_parameters=include_nuisance_parameters
        ):
            # Cuts
            cut_filter = [self._pass_cuts(obs_event, cuts) for obs_event in observations]
            observations = observations[cut_filter]
            weights = weights[cut_filter]

            # Efficiencies
            efficiencies = np.array(
                [self._eval_efficiency(obs_event, efficiency_functions) for obs_event in observations]
            )
            weights *= efficiencies[:, np.newaxis]

            # xsecs
            if xsecs_benchmarks is None:
                xsecs_benchmarks = np.sum(weights, axis=0)
                xsecs_uncertainty_benchmarks = np.sum(weights ** 2, axis=0)
            else:
                xsecs_benchmarks += np.sum(weights, axis=0)
                xsecs_uncertainty_benchmarks += np.sum(weights ** 2, axis=0)

        assert xsecs_benchmarks is not None, "No events passed cuts"

        xsecs_uncertainty_benchmarks = xsecs_uncertainty_benchmarks ** 0.5

        logger.debug("Benchmarks xsecs [pb]: %s", xsecs_benchmarks)

        if return_benchmark_xsecs:
            if return_error:
                return xsecs_benchmarks, xsecs_uncertainty_benchmarks
            return xsecs_benchmarks

        # Translate to xsec for theta
        theta_matrix = get_theta_benchmark_matrix("morphing", theta, self.benchmarks, self.morpher)
        xsec = mdot(theta_matrix, xsecs_benchmarks)
        xsec_error = mdot(theta_matrix, xsecs_uncertainty_benchmarks)

        logger.debug("Theta matrix: %s", theta_matrix)
        logger.debug("Cross section at theta: %s pb", xsec)

        if return_error:
            return xsec, xsec_error
        return xsec

    def _calculate_xsecs_limits(self, thetas, test_split=0.2):
        # Test split
        start_event, end_event = self._train_test_split(False, test_split)

        # Total xsecs for benchmarks
        xsecs_benchmarks = 0.0
        for observations, weights in madminer_event_loader(self.madminer_filename, start=start_event, end=end_event):
            xsecs_benchmarks += np.sum(weights, axis=0)

        # xsecs at thetas
        xsecs = []
        for theta in thetas:
            theta_matrix = get_theta_benchmark_matrix("morphing", theta, self.benchmarks, self.morpher)
            xsecs.append(mdot(theta_matrix, xsecs_benchmarks))
        return np.asarray(xsecs)

    def _train_test_split(self, train, test_split):
        """
        Returns the start and end event for train samples (train = True) or test samples (train = False).

        Parameters
        ----------
        train : bool
            True if training data is generated, False if test data is generated.

        test_split : float
            Fraction of events reserved for testing.

        Returns
        -------
        start_event : int
            Index (in the MadMiner file) of the first event to consider.

        end_event : int
            Index (in the MadMiner file) of the last unweighted event to consider.

        """
        if train:
            start_event = 0

            if test_split is None or test_split <= 0.0 or test_split >= 1.0:
                end_event = None
            else:
                end_event = int(round((1.0 - test_split) * self.n_samples, 0))
                if end_event < 0 or end_event > self.n_samples:
                    raise ValueError("Irregular train / test split: sample {} / {}", end_event, self.n_samples)

        else:
            if test_split is None or test_split <= 0.0 or test_split >= 1.0:
                start_event = 0
            else:
                start_event = int(round((1.0 - test_split) * self.n_samples, 0)) + 1
                if start_event < 0 or start_event > self.n_samples:
                    raise ValueError("Irregular train / test split: sample {} / {}", start_event, self.n_samples)

            end_event = None

        return start_event, end_event
