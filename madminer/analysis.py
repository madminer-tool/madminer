from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
import collections
import six

from madminer.utils.interfaces.madminer_hdf5 import load_madminer_settings, madminer_event_loader
from madminer.utils.interfaces.madminer_hdf5 import save_preformatted_events_to_madminer_file
from madminer.utils.analysis import get_theta_value, get_theta_benchmark_matrix, get_dtheta_benchmark_matrix
from madminer.utils.analysis import get_nu_value
from madminer.utils.analysis import calculate_augmented_data, parse_theta, mdot
from madminer.utils.morphing import PhysicsMorpher, NuisanceMorpher
from madminer.utils.various import format_benchmark, create_missing_folders, shuffle, balance_thetas

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

    def extract_raw_data(self, theta=None, derivative=False):

        """
        Returns all events together with the benchmark weights (if theta is None) or weights for a given theta.

        Parameters
        ----------
        theta : None or ndarray or str, optional
            If None, the function returns all benchmark weights. If str, the function returns the weights for a given
            benchmark name. If ndarray, it uses morphing to calculate the weights for this value of theta. Default
            value: None.

        derivative : bool, optional
            If True and if theta is not None, the derivative of the weights with respect to theta are returned. Default
            value: False.

        Returns
        -------
        x : ndarray
            Observables with shape `(n_unweighted_samples, n_observables)`.

        weights : ndarray
            If theta is None and derivative is False, benchmark weights with shape
            `(n_unweighted_samples, n_benchmarks_phys)` in pb. If theta is not None and derivative is True, the gradient of
            the weight for the given parameter with respect to theta with shape `(n_unweighted_samples, n_gradients)`
            in pb. Otherwise, weights for the given parameter theta with shape `(n_unweighted_samples,)` in pb.

        """

        x, weights_benchmarks = next(madminer_event_loader(self.madminer_filename, batch_size=None))

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
            theta_matrix = get_theta_benchmark_matrix("morphing", theta, self.benchmarks, self.morpher)

            weights_theta = mdot(theta_matrix, weights_benchmarks)

            return x, weights_theta

    def _calculate_benchmark_xsecs(self, start_event, end_event, use_nuisance_parameters):
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
            Index of the first unweighted event to consider.

        end_event : int
            Index of the last unweighted event to consider.

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
