import logging
import numpy as np

from madminer.utils.interfaces.hdf5 import load_events
from madminer.utils.interfaces.hdf5 import load_madminer_settings
from madminer.utils.morphing import PhysicsMorpher, NuisanceMorpher
from madminer.utils.various import mdot

logger = logging.getLogger(__name__)


class DataAnalyzer:
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

        # Load data
        logger.info("Loading data from %s", filename)
        (
            self.parameters,
            self.benchmarks,
            self.benchmark_nuisance_flags,
            self.morphing_components,
            self.morphing_matrix,
            self.observables,
            self.n_samples,
            self.systematics,
            self.reference_benchmark,
            self.nuisance_parameters,
            self.n_events_generated_per_benchmark,
            self.n_events_backgrounds,
            self.finite_difference_benchmarks,
            self.finite_difference_epsilon,
        ) = load_madminer_settings(filename, include_nuisance_benchmarks=include_nuisance_parameters)

        self.n_observables = len(self.observables)
        self.n_parameters = len(self.parameters)
        self.n_benchmarks = len(self.benchmarks)
        self.n_benchmarks_phys = np.sum(np.logical_not(self.benchmark_nuisance_flags))
        self.n_nuisance_parameters = len(self.nuisance_parameters)

        # Morphing
        self.morpher = None
        if self.morphing_matrix is not None and self.morphing_components is not None and not disable_morphing:
            self.morpher = PhysicsMorpher(self.parameters)
            self.morpher.set_components(self.morphing_components)
            self.morpher.set_basis(self.benchmarks, morphing_matrix=self.morphing_matrix)

        # Nuisance morphing
        self.nuisance_morpher = None
        if self.n_nuisance_parameters > 0:
            self.nuisance_morpher = NuisanceMorpher(
                self.nuisance_parameters,
                self.benchmarks.keys(),
                self.reference_benchmark,
            )

        # Check event numbers
        self._check_n_events()
        self._report_setup()

    def event_loader(
        self,
        start=0,
        end=None,
        batch_size=100000,
        include_nuisance_parameters=None,
        generated_close_to=None,
        return_sampling_ids=False,
    ):
        """
        Yields batches of events in the MadMiner file.

        Parameters
        ----------
        start : int, optional
            First event index to load

        end : int or None, optional
            Last event index to load

        batch_size : int, optional
            Batch size

        include_nuisance_parameters : bool, optional
            Whether nuisance parameter benchmarks are included in the returned data

        generated_close_to : None or ndarray, optional
            If None, this function yields all events. Otherwise, it just yields just the events that were generated
            at the closest benchmark point to a given parameter point.

        return_sampling_ids : bool, optional
            If True, the iterator returns the sampling IDs in addition to observables and weights.

        Yields
        ------
        observations : ndarray
            Event data

        weights : ndarray
            Event weights

        sampling_ids : int
            Sampling IDs (benchmark used for sampling for signal events, -1 for background events). Only returned if
            return_sampling_ids = True was set.
        """

        if include_nuisance_parameters is None:
            include_nuisance_parameters = self.include_nuisance_parameters

        sampling_benchmark = self._find_closest_benchmark(generated_close_to)
        logger.debug(f"Sampling benchmark closest to {generated_close_to}: {sampling_benchmark}")

        if sampling_benchmark is None:
            sampling_factors = self._calculate_sampling_factors()
        else:
            sampling_factors = np.ones(self.n_benchmarks_phys + 1)

        for data in load_events(
            file_name=self.madminer_filename,
            start_index=start,
            final_index=end,
            batch_size=batch_size,
            benchmark_nuisance_flags=self.benchmark_nuisance_flags,
            sampling_benchmark=sampling_benchmark,
            sampling_factors=sampling_factors,
            include_nuisance_params=include_nuisance_parameters,
            include_sampling_ids=return_sampling_ids,
        ):
            yield data

    def weighted_events(
        self,
        theta=None,
        nu=None,
        start_event=None,
        end_event=None,
        derivative=False,
        generated_close_to=None,
        n_draws=None,
    ):
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

        generated_close_to : None or int, optional
            Only returns benchmarks generated from this benchmark (and background events). Default value: None.

        n_draws : None or int, optional
            If not None, returns only this number of events, drawn randomly.

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
            self.event_loader(
                start=start_event,
                end=end_event,
                batch_size=None,
                generated_close_to=generated_close_to,
            )
        )

        # Pick events randomly
        n_events = len(x)

        if n_draws is not None and n_draws < n_events:
            idx = np.random.choice(n_events, n_draws, replace=False)
            x = x[idx]
            weights_benchmarks = weights_benchmarks[idx]
        elif n_draws is not None:
            logger.warning(f"Requested {n_draws} events, but only {n_events} available")

        # Process and return appropriate weights
        if theta is None:
            return x, weights_benchmarks
        elif isinstance(theta, str):
            i_benchmark = list(self.benchmarks.keys()).index(theta)
            return x, weights_benchmarks[:, i_benchmark]
        elif derivative:
            dtheta_matrix = self._get_dtheta_benchmark_matrix(theta)
            gradients_theta = mdot(dtheta_matrix, weights_benchmarks)  # (n_gradients, n_samples)
            gradients_theta = gradients_theta.T
            return x, gradients_theta
        else:
            # TODO: nuisance params
            if nu is not None:
                raise NotImplementedError()
            theta_matrix = self._get_theta_benchmark_matrix(theta)
            weights_theta = mdot(theta_matrix, weights_benchmarks)
            return x, weights_theta

    def xsecs(
        self,
        thetas=None,
        nus=None,
        partition="all",
        test_split=0.2,
        validation_split=0.2,
        include_nuisance_benchmarks=True,
        batch_size=100000,
        generated_close_to=None,
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

        partition : {"train", "test", "validation", "all"}, optional
            Which event partition to use. Default: "all".

        test_split : float, optional
            Fraction of events reserved for testing. Default value: 0.2.

        validation_split : float, optional
            Fraction of weighted events reserved for validation. Default value: 0.2.

        include_nuisance_benchmarks : bool, optional
            Whether to include nuisance benchmarks if thetas is None. Default value: True.

        batch_size : int, optional
            Size of the batches of events that are loaded into memory at the same time. Default value: 100000.

        generated_close_to : None or ndarray, optional
            If not None, only events originally generated from the closest benchmark to this parameter point will be
            used. Default value : None.

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
            include_nuisance_benchmarks = True
        if thetas is not None:
            if nus is None:
                nus = [None for _ in thetas]
            assert len(nus) == len(thetas), "Numbers of thetas and nus don't match!"

        # Which events to use
        if partition == "all":
            start_event, end_event = None, None
            correction_factor = 1.0
        elif partition in ["train", "validation", "test"]:
            start_event, end_event, correction_factor = self._train_validation_test_split(
                partition, test_split, validation_split
            )
        else:
            raise ValueError(f"Invalid partition type: {partition}")

        # Theta matrices (translation of benchmarks to theta, at nominal nuisance params)
        if thetas is None:
            theta_matrices = np.identity(self.n_benchmarks)
        else:
            theta_matrices = [self._get_theta_benchmark_matrix(theta) for theta in thetas]
            theta_matrices = np.asarray(theta_matrices)  # Shape (n_thetas, n_benchmarks)

        # Loop over events
        xsecs = 0.0
        xsec_uncertainties = 0.0
        n_events = 0

        for i_batch, (_, benchmark_weights) in enumerate(
            self.event_loader(
                start=start_event,
                end=end_event,
                include_nuisance_parameters=include_nuisance_benchmarks,
                batch_size=batch_size,
                generated_close_to=generated_close_to,
            )
        ):
            n_batch, _ = benchmark_weights.shape
            n_events += n_batch

            # Benchmark xsecs
            if thetas is None:
                xsecs += np.sum(benchmark_weights, axis=0)
                xsec_uncertainties += np.sum(benchmark_weights * benchmark_weights, axis=0)

            # xsecs at given parameters(theta, nu)
            else:
                # Weights at nominal nuisance params (nu=0)
                weights_nom = mdot(theta_matrices, benchmark_weights)  # Shape (n_thetas, n_batch)
                weights_sq_nom = mdot(theta_matrices, benchmark_weights * benchmark_weights)  # same

                # Effect of nuisance parameters
                nuisance_factors = self._calculate_nuisance_factors(nus, benchmark_weights)
                weights = nuisance_factors * weights_nom
                weights_sq = nuisance_factors * weights_sq_nom

                # Sum up
                xsecs += np.sum(weights, axis=1)
                xsec_uncertainties += np.sum(weights_sq, axis=1)

        if n_events == 0:
            raise RuntimeError(
                f"Did not find events with test_split = {test_split} "
                f"and generated_close_to = {generated_close_to}"
            )

        xsec_uncertainties = np.maximum(xsec_uncertainties, 0.0) ** 0.5

        # Correct for not using all events
        xsecs *= correction_factor
        xsec_uncertainties *= correction_factor

        logger.debug("xsecs and uncertainties [pb]:")
        for this_xsec, this_uncertainty in zip(xsecs, xsec_uncertainties):
            logger.debug("  (%4f +/- %4f) pb (%4f %%)", this_xsec, this_uncertainty, 100 * this_uncertainty / this_xsec)

        return xsecs, xsec_uncertainties

    def xsec_gradients(
        self,
        thetas,
        nus=None,
        partition="all",
        test_split=0.2,
        validation_split=0.2,
        gradients="all",
        batch_size=100000,
        generated_close_to=None,
    ):
        """
        Returns the gradient of total cross sections with respect to parameters.

        Parameters
        ----------
        thetas : list of (ndarray or str), optional
            If None, the function returns all benchmark cross sections. Otherwise, it returns the cross sections for a
            series of parameter points that are either given by their benchmark name (as a str), their benchmark index
            (as an int), or their parameter value (as an ndarray, using morphing). Default value: None.

        nus : None or list of (None or ndarray), optional
             If None, the nuisance parameters are set to their nominal values (0), i.e. no systematics are taken into
             account. Otherwise, the list has to have the same number of elements as thetas, and each entry can specify
             nuisance parameters at nominal value (None) or a value of the nuisance parameters (ndarray).

        partition : {"train", "test", "validation", "all"}, optional
            Which events to use. Default: "all".

        test_split : float, optional
            Fraction of events reserved for testing. Default value: 0.2.

        validation_split : float, optional
            Fraction of weighted events reserved for validation. Default value: 0.2.

        gradients : {"all", "theta", "nu"}, optional
            Which gradients to calculate. Default value: "all".

        batch_size : int, optional
            Size of the batches of events that are loaded into memory at the same time. Default value: 100000.

        generated_close_to : None or ndarray, optional
            If not None, only events originally generated from the closest benchmark to this parameter point will be
            used. Default value : None.

        Returns
        -------
        xsecs_gradients : ndarray
            Calculated cross section gradients in pb with shape (n_gradients,).
        """

        logger.debug(f"Calculating cross section gradients for thetas = {thetas} and nus = {nus}")

        # Inputs
        include_nuisance_benchmarks = nus is not None or gradients in ["all", "nu"]
        if nus is None:
            nus = [None for _ in thetas]
        assert len(nus) == len(thetas), "Numbers of thetas and nus don't match!"
        if gradients not in ["all", "theta", "nu"]:
            raise RuntimeError(f"Invalid gradients type: {gradients}")

        # Which events to use
        if partition == "all":
            start_event, end_event = None, None
            correction_factor = 1.0
        elif partition in ["train", "validation", "test"]:
            start_event, end_event, correction_factor = self._train_validation_test_split(
                partition, test_split, validation_split
            )
        else:
            raise ValueError(f"Invalid partition type: {partition}")

        # Theta matrices (translation of benchmarks to theta, at nominal nuisance params)
        theta_matrices = np.asarray(
            [self._get_theta_benchmark_matrix(theta) for theta in thetas]
        )  # shape (n_thetas, n_benchmarks)

        theta_gradient_matrices = np.asarray(
            [self._get_dtheta_benchmark_matrix(theta) for theta in thetas]
        )  # shape (n_thetas, n_gradients, n_benchmarks)

        # Loop over events
        xsec_gradients = 0.0

        for i_batch, (_, benchmark_weights) in enumerate(
            self.event_loader(
                start=start_event,
                end=end_event,
                include_nuisance_parameters=include_nuisance_benchmarks,
                batch_size=batch_size,
                generated_close_to=generated_close_to,
            )
        ):
            n_batch, _ = benchmark_weights.shape
            logger.debug(f"Batch {i_batch+1} with {n_batch} events")

            if gradients in ["all", "theta"]:
                nom_gradients = mdot(
                    theta_gradient_matrices, benchmark_weights
                )  # Shape (n_thetas, n_phys_gradients, n_batch)
                nuisance_factors = self._calculate_nuisance_factors(nus, benchmark_weights)  # Shape (n_thetas, n_batch)
                try:
                    dweight_dtheta = nuisance_factors[:, np.newaxis, :] * nom_gradients
                except TypeError:
                    dweight_dtheta = nom_gradients

            if gradients in ["all", "nu"]:
                weights_nom = mdot(theta_matrices, benchmark_weights)  # Shape (n_thetas, n_batch)
                nuisance_factor_gradients = np.asarray(
                    [self.nuisance_morpher.calculate_nuisance_factor_gradients(nu, benchmark_weights) for nu in nus]
                )  # Shape (n_thetas, n_nuisance_gradients, n_batch)
                dweight_dnu = nuisance_factor_gradients * weights_nom[:, np.newaxis, :]

            if gradients == "all":
                dweight_dall = np.concatenate((dweight_dtheta, dweight_dnu), 1)
            elif gradients == "theta":
                dweight_dall = dweight_dtheta
            elif gradients == "nu":
                dweight_dall = dweight_dnu
            xsec_gradients += np.sum(dweight_dall, axis=2)

        # Correct for not using all events
        xsec_gradients *= correction_factor

        return xsec_gradients

    def _check_n_events(self):
        n_events_check = \
            sum(self.n_events_generated_per_benchmark) \
            + self.n_events_backgrounds

        if self.n_samples != n_events_check:
            logger.warning(
                "Inconsistent event numbers in HDF5 file! Please recalculate them by calling "
                "combine_and_shuffle(recalculate_header=True)."
            )

    def _report_setup(self):
        logger.info(f"Found {self.n_parameters} parameters")
        for i, param in enumerate(self.parameters.values()):
            logger.info("  %s: %s", i, param)

        if self.n_nuisance_parameters > 0:
            logger.info(f"Found {self.n_nuisance_parameters} nuisance parameters")
            for i, param in enumerate(self.nuisance_parameters.values()):
                logger.info("  %s: %s", i, param)
        else:
            logger.info("Did not find nuisance parameters")

        logger.info(f"Found {self.n_benchmarks} benchmarks")
        for benchmark in self.benchmarks.values():
            if benchmark.is_nuisance:
                logger.debug("   %s: systematics", benchmark.name)
            else:
                logger.debug("   %s", benchmark)

        logger.info(f"Found {self.n_observables} observables")
        for i, obs in enumerate(self.observables):
            logger.debug("  %2.2s %s", i, obs)

        logger.info(f"Found {self.n_samples} events")
        if len(self.n_events_generated_per_benchmark) > 0:
            for events, name in zip(self.n_events_generated_per_benchmark, self.benchmarks.keys()):
                logger.info("  %s signal events sampled from benchmark %s", events, name)
            if self.n_events_backgrounds is not None and self.n_events_backgrounds > 0:
                logger.info("  %s background events", self.n_events_backgrounds)
        else:
            logger.debug("  Did not find sample summary information")

        if self.morpher is not None:
            logger.info("Found morphing setup with %s components", len(self.morphing_components))
        else:
            logger.info("Did not find morphing setup.")

        if self.nuisance_morpher is not None:
            logger.info("Found nuisance morphing setup")
        else:
            logger.info("Did not find nuisance morphing setup")

    def _calculate_nuisance_factors(self, nus, benchmark_weights):
        if self._any_nontrivial_nus(nus):
            return np.asarray(
                [self.nuisance_morpher.calculate_nuisance_factors(nu, benchmark_weights) for nu in nus]
            )  # Shape (n_thetas, n_batch)
        else:
            return 1.0

    def _finite_differences_theta_gradient_matrices(self):
        """ Constructs the matrix that translates benchmark weights to the gradient of the weight evaluated at the benchmarks """
        assert self.finite_difference_benchmarks is not None
        assert self.finite_difference_epsilon is not None

        matrix = np.zeros(
            (self.n_benchmarks, self.n_parameters, self.n_benchmarks)
        )  # (n_thetas, n_gradients, n_benchmarks)

        benchmark_names = list(self.benchmarks.keys())

        # We'll generally try to find the tuples p, i, j, k such that
        # matrix[i, p, j] = - 1 / eps and matrix[i, p, i] = 1 / eps
        for i, b_name in enumerate(self.benchmarks.keys()):
            # For the FD-shifted benchmarks, we assume that the gradients are
            # the same as at the original point, and will just copy the matrix later
            copy_to = []
            if b_name not in self.finite_difference_benchmarks.keys():
                continue

            for p, p_name in enumerate(self.parameters.keys()):
                shifted_benchmark_dict = self.finite_difference_benchmarks[b_name].shift_names
                shifted_benchmark_name = shifted_benchmark_dict[p_name]
                j = benchmark_names.index(shifted_benchmark_name)
                copy_to.append(j)

                matrix[i, p, j] = +1.0 / self.finite_difference_epsilon
                matrix[i, p, i] = -1.0 / self.finite_difference_epsilon

            for j in copy_to:
                matrix[j, :, :] = matrix[i, :, :]

        return matrix

    @staticmethod
    def _any_nontrivial_nus(nus):
        if nus is None:
            return False
        for nu in nus:
            if nu is not None:
                return True
        return False

    def _derivative_mode(self):
        if self.morpher is not None:
            mode = "morphing"
        elif self.finite_difference_benchmarks is not None:
            mode = "fd"
        else:
            raise RuntimeError(
                "Cannot compute xsec gradients when neither morphing nor finite differences are correctly set up!"
            )
        return mode

    def _weights(self, thetas, nus, benchmark_weights, theta_matrices=None):
        """
        Turns benchmark weights into weights for given parameter points (theta, nu).

        Parameters
        ----------
        thetas : list of (ndarray or str)
            If None, the function returns all benchmark cross sections. Otherwise, it returns the cross sections for a
            series of parameter points that are either given by their benchmark name (as a str), their benchmark index
            (as an int), or their parameter value (as an ndarray, using morphing).

        nus : None or list of (None or ndarray)
             If None, the nuisance parameters are set to their nominal values (0), i.e. no systematics are taken into
             account. Otherwise, the list has to have the same number of elements as thetas, and each entry can specify
             nuisance parameters at nominal value (None) or a value of the nuisance parameters (ndarray).

        Returns
        -------
        weights : ndarray
            Calculated weights in pb.
        """

        n_events, _ = benchmark_weights.shape

        # Inputs
        if nus is None:
            nus = [None for _ in thetas]
        assert len(nus) == len(thetas), "Numbers of thetas and nus don't match!"

        # Theta matrices (translation of benchmarks to theta, at nominal nuisance params)
        if theta_matrices is None:
            theta_matrices = [self._get_theta_benchmark_matrix(theta) for theta in thetas]
        theta_matrices = np.asarray(theta_matrices)  # Shape (n_thetas, n_benchmarks)

        # Weights at nominal nuisance params (nu=0)
        weights_nom = mdot(theta_matrices, benchmark_weights)  # Shape (n_thetas, n_batch)

        # Effect of nuisance parameters
        nuisance_factors = self._calculate_nuisance_factors(nus, benchmark_weights)
        weights = nuisance_factors * weights_nom

        return weights

    def _weight_gradients(
        self, thetas, nus, benchmark_weights, gradients="all", theta_matrices=None, theta_gradient_matrices=None
    ):
        """
        Turns benchmark weights into weights for given parameter points (theta, nu).

        Parameters
        ----------
        thetas : list of (ndarray or str)
            If None, the function returns all benchmark cross sections. Otherwise, it returns the cross sections for a
            series of parameter points that are either given by their benchmark name (as a str), their benchmark index
            (as an int), or their parameter value (as an ndarray, using morphing).

        nus : None or list of (None or ndarray)
             If None, the nuisance parameters are set to their nominal values (0), i.e. no systematics are taken into
             account. Otherwise, the list has to have the same number of elements as thetas, and each entry can specify
             nuisance parameters at nominal value (None) or a value of the nuisance parameters (ndarray).

        gradients : {"all", "theta", "nu"}, optional
            Which gradients to calculate. Default value: "all".

        Returns
        -------
        gradients : ndarray
            Calculated gradients in pb.
        """

        n_events, _ = benchmark_weights.shape

        # Inputs
        if gradients == "all" and self.n_nuisance_parameters == 0:
            gradients = "theta"
        if nus is None:
            nus = [None for _ in thetas]
        assert len(nus) == len(thetas), "Numbers of thetas and nus don't match!"

        # Theta matrices (translation of benchmarks to theta, at nominal nuisance params)
        if theta_matrices is None:
            theta_matrices = [self._get_theta_benchmark_matrix(theta) for theta in thetas]
        if theta_gradient_matrices is None:
            theta_gradient_matrices = [self._get_dtheta_benchmark_matrix(theta) for theta in thetas]
        theta_matrices = np.asarray(theta_matrices)  # Shape (n_thetas, n_benchmarks)
        theta_gradient_matrices = np.asarray(theta_gradient_matrices)  # Shape (n_thetas, n_gradients, n_benchmarks)

        # Calculate theta gradient
        if gradients in ["all", "theta"]:
            nom_gradients = mdot(theta_gradient_matrices, benchmark_weights)  # (n_thetas, n_phys_gradients, n_batch)
            nuisance_factors = self._calculate_nuisance_factors(nus, benchmark_weights)
            try:
                dweight_dtheta = nuisance_factors[:, np.newaxis, :] * nom_gradients
            except TypeError:
                dweight_dtheta = nom_gradients
        else:
            dweight_dtheta = None

        # Calculate nu gradient
        if gradients in ["all", "nu"]:
            weights_nom = mdot(theta_matrices, benchmark_weights)  # Shape (n_thetas, n_batch)
            nuisance_factor_gradients = np.asarray(
                [self.nuisance_morpher.calculate_nuisance_factor_gradients(nu, benchmark_weights) for nu in nus]
            )  # Shape (n_thetas, n_nuisance_gradients, n_batch)
            dweight_dnu = nuisance_factor_gradients * weights_nom[:, np.newaxis, :]
        else:
            dweight_dnu = None

        if gradients == "theta":
            return dweight_dtheta
        elif gradients == "nu":
            return dweight_dnu
        return np.concatenate((dweight_dtheta, dweight_dnu), 1)

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

        correction_factor : float
            Factor with which the weights and cross sections will have to be multiplied to make up for the missing
            events.

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
                    raise ValueError(f"Irregular split: sample {end_event} / {self.n_samples}")

        else:
            if test_split is None or test_split <= 0.0 or test_split >= 1.0:
                start_event = 0
                correction_factor = 1.0
            else:
                start_event = int(round((1.0 - test_split) * self.n_samples, 0)) + 1
                correction_factor = 1.0 / test_split
                if start_event < 0 or start_event > self.n_samples:
                    raise ValueError(f"Irregular split: sample {start_event} / {self.n_samples}")

            end_event = None

        return start_event, end_event, correction_factor

    def _train_validation_test_split(self, partition, test_split, validation_split):
        """
        Returns the start and end event for train samples (train = True) or test samples (train = False).

        Parameters
        ----------
        partition : ["train", "validation", "test"]

        test_split : float
            Fraction of events reserved for testing.

        validation_split : float
            Fraction of events reserved for testing.

        Returns
        -------
        start_event : int
            Index (in the MadMiner file) of the first event to consider.

        end_event : int
            Index (in the MadMiner file) of the last unweighted event to consider.

        correction_factor : float
            Factor with which the weights and cross sections will have to be multiplied to make up for the missing
            events.

        """
        if test_split is None or test_split < 0.0:
            test_split = 0.0
        if validation_split is None or validation_split < 0.0:
            validation_split = 0.0
        assert test_split + validation_split <= 1.0
        train_split = 1.0 - test_split - validation_split

        if partition == "train":
            start_event = 0

            if test_split is None or test_split <= 0.0 or test_split >= 1.0:
                end_event = None
                correction_factor = 1.0
            else:
                end_event = int(round(train_split * self.n_samples, 0))
                correction_factor = 1.0 / train_split

                if end_event < 0 or end_event > self.n_samples:
                    raise ValueError(f"Irregular split: sample {end_event} / {self.n_samples}")

        elif partition == "validation":
            if validation_split is None or validation_split <= 0.0 or validation_split >= 1.0:
                start_event = 0
                end_event = None
                correction_factor = 1.0

            else:
                start_event = int(round(train_split * self.n_samples, 0)) + 1
                end_event = int(round((1.0 - test_split) * self.n_samples, 0))
                correction_factor = 1.0 / validation_split

                if start_event < 0 or start_event > self.n_samples:
                    raise ValueError(f"Irregular split: sample {start_event} / {self.n_samples}")

                if end_event < 0 or end_event > self.n_samples:
                    raise ValueError(f"Irregular split: sample {end_event} / {self.n_samples}")

        elif partition == "test":
            end_event = None

            if test_split is None or test_split <= 0.0 or test_split >= 1.0:
                start_event = 0
                correction_factor = 1.0
            else:
                start_event = int(round((1.0 - test_split) * self.n_samples, 0)) + 1
                correction_factor = 1.0 / test_split
                if start_event < 0 or start_event > self.n_samples:
                    raise ValueError(f"Irregular split: sample {start_event} / {self.n_samples}")

        else:
            raise RuntimeError(f"Unknown partition {partition}")

        return start_event, end_event, correction_factor

    def _get_theta_value(self, theta):
        if isinstance(theta, str):
            benchmark = self.benchmarks[theta]
            theta_value = np.array([val for val in benchmark.values.values()])
        elif isinstance(theta, int):
            benchmark = self.benchmarks[list(self.benchmarks.keys())[theta]]
            theta_value = np.array([val for val in benchmark.values.values()])
        else:
            theta_value = np.asarray(theta)
        return theta_value

    def _get_nu_value(self, nu):
        if nu is None:
            nu_value = np.zeros(self.n_nuisance_parameters)
        else:
            nu_value = np.asarray(nu)
        return nu_value

    def _get_theta_benchmark_matrix(self, theta, zero_pad=True):
        """Calculates vector A such that dsigma(theta) = A * dsigma_benchmarks"""

        if zero_pad:
            unpadded_theta_matrix = self._get_theta_benchmark_matrix(theta, zero_pad=False)
            theta_matrix = np.zeros(self.n_benchmarks)
            theta_matrix[: unpadded_theta_matrix.shape[0]] = unpadded_theta_matrix

        elif isinstance(theta, str):
            i_benchmark = list(self.benchmarks).index(theta)
            theta_matrix = self._get_theta_benchmark_matrix(i_benchmark)

        elif isinstance(theta, int):
            n_benchmarks = len(self.benchmarks)
            theta_matrix = np.zeros(n_benchmarks)
            theta_matrix[theta] = 1.0

        else:
            theta_matrix = self.morpher.calculate_morphing_weights(theta)

        return theta_matrix

    def _get_dtheta_benchmark_matrix(self, theta, zero_pad=True):
        """Calculates matrix A_ij such that d dsigma(theta) / d theta_i = A_ij * dsigma (benchmark j)"""

        mode = self._derivative_mode()

        if zero_pad:
            unpadded_theta_matrix = self._get_dtheta_benchmark_matrix(theta, zero_pad=False)
            dtheta_matrix = np.zeros((unpadded_theta_matrix.shape[0], self.n_benchmarks))
            dtheta_matrix[:, : unpadded_theta_matrix.shape[1]] = unpadded_theta_matrix

        elif isinstance(theta, str) and mode == "morphing":
            benchmark = self.benchmarks[theta]
            benchmark = np.array([val for val in benchmark.values.values()])
            dtheta_matrix = self._get_dtheta_benchmark_matrix(benchmark)

        elif isinstance(theta, int) and mode == "morphing":
            benchmark = self.benchmarks[list(self.benchmarks.keys())[theta]]
            benchmark = np.array([val for val in benchmark.values.values()])
            dtheta_matrix = self._get_dtheta_benchmark_matrix(benchmark)

        elif isinstance(theta, str):
            benchmark_id = list(self.benchmarks.keys()).index(theta)
            dtheta_matrix = self._get_dtheta_benchmark_matrix(benchmark_id)

        elif isinstance(theta, int):  # finite differences
            # TODO: avoid constructing the full matrix every time
            dtheta_matrix = self._finite_differences_theta_gradient_matrices()[theta]

        else:
            if mode == "fd":
                raise RuntimeError("Cannot calculate score for arbitrary parameter points without morphing setup")

            # Shape (n_parameters, n_benchmarks_phys)
            dtheta_matrix = self.morpher.calculate_morphing_weight_gradient(theta)

        return dtheta_matrix

    def _calculate_sampling_factors(self):
        events = np.asarray(self.n_events_generated_per_benchmark, dtype=np.float64)
        logger.debug(f"Events per benchmark: {events}")
        factors = events / np.sum(events)
        factors = np.hstack((factors, 1.0))  # background events
        return factors

    def _find_closest_benchmark(self, theta):
        if theta is None:
            return None

        benchmarks = self._benchmark_array()[: self.n_benchmarks_phys]
        distances = [np.linalg.norm(benchmark - theta) for benchmark in benchmarks]

        # Don't use benchmarks where we don't actually have events
        if len(self.n_events_generated_per_benchmark) > 0:
            distances = distances + 1.0e9 * (self.n_events_generated_per_benchmark == 0).astype(np.float64)

        closest_idx = np.argmin(distances)
        return closest_idx

    def _benchmark_array(self):
        return np.asarray([
            list(benchmark.values.values()) for benchmark in self.benchmarks.values()
        ])
