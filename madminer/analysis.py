from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
import collections
import six

from madminer.utils.interfaces.madminer_hdf5 import load_madminer_settings, madminer_event_loader
from madminer.utils.analysis import _get_theta_benchmark_matrix, _get_dtheta_benchmark_matrix
from madminer.utils.analysis import mdot
from madminer.utils.morphing import PhysicsMorpher, NuisanceMorpher
from madminer.utils.various import format_benchmark

logger = logging.getLogger(__name__)


class DataAnalyzer(object):
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
            dtheta_matrix = _get_dtheta_benchmark_matrix("morphing", theta, self.benchmarks, self.morpher)

            gradients_theta = mdot(dtheta_matrix, weights_benchmarks)  # (n_gradients, n_samples)
            gradients_theta = gradients_theta.T

            return x, gradients_theta

        else:
            # TODO: nuisance params
            if nu is not None:
                raise NotImplementedError

            theta_matrix = _get_theta_benchmark_matrix("morphing", theta, self.benchmarks, self.morpher)

            weights_theta = mdot(theta_matrix, weights_benchmarks)

            return x, weights_theta

    def xsecs(
        self, thetas=None, nus=None, events="all", test_split=0.2, include_nuisance_benchmarks=False, batch_size=100000
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

        include_nuisance_benchmarks : bool, optional
            Whether to includee nuisance benchmarks if thetas is None. Default value: False.

        test_split : float, optional
            Fraction of events reserved for testing. Default value: 0.2.

        events : {"train", "test", "all"}, optional
            Which events to use. Default: "all".

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

        # Which events to use
        if events == "all":
            start_event, end_event = None, None
            correction_factor = 1.0
        elif events == "train":
            start_event, end_event, correction_factor = self._train_test_split(True, test_split)
        elif events == "test":
            start_event, end_event, correction_factor = self._train_test_split(False, test_split)
        else:
            raise ValueError("Events has to be either 'all', 'train', or 'test', but got {}!".format(events))

        # Theta matrices (translation of benchmarks to theta, at nominal nuisance params)
        theta_matrices = []
        for theta in thetas:
            if isinstance(theta, six.string_types):
                i_benchmark = list(self.benchmarks.keys()).index(theta)
                theta_matrix = _get_theta_benchmark_matrix("benchmark", i_benchmark, self.benchmarks)
            elif isinstance(theta, int):
                theta_matrix = _get_theta_benchmark_matrix("benchmark", theta, self.benchmarks)
            else:
                theta_matrix = _get_theta_benchmark_matrix("morphing", theta, self.benchmarks, self.morpher)
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

        # Correct for not using all events
        xsecs *= correction_factor
        xsec_uncertainties *= correction_factor

        logger.debug("xsecs and uncertainties [pb]:")
        for this_xsec, this_uncertainty in zip(xsecs, xsec_uncertainties):
            logger.debug("  %s +/- %s (%s %)", this_xsec, this_uncertainty, 100 * this_uncertainty / this_xsec)

        return xsecs, xsec_uncertainties

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
                correction_factor = 1.0
            else:
                end_event = int(round((1.0 - test_split) * self.n_samples, 0))
                correction_factor = 1.0 / (1.0 - test_split)
                if end_event < 0 or end_event > self.n_samples:
                    raise ValueError("Irregular train / test split: sample {} / {}", end_event, self.n_samples)

        else:
            if test_split is None or test_split <= 0.0 or test_split >= 1.0:
                start_event = 0
                correction_factor = 1.0
            else:
                start_event = int(round((1.0 - test_split) * self.n_samples, 0)) + 1
                correction_factor = 1.0 / (test_split)
                if start_event < 0 or start_event > self.n_samples:
                    raise ValueError("Irregular train / test split: sample {} / {}", start_event, self.n_samples)

            end_event = None

        return start_event, end_event, correction_factor
