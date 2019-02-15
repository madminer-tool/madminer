from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
import collections
import six

from madminer.utils.interfaces.madminer_hdf5 import load_madminer_settings, madminer_event_loader
from madminer.utils.interfaces.madminer_hdf5 import save_preformatted_events_to_madminer_file
from madminer.utils.analysis import get_theta_value, get_theta_benchmark_matrix, get_dtheta_benchmark_matrix
from madminer.utils.analysis import calculate_augmented_data, parse_theta, mdot
from madminer.morphing import Morpher, NuisanceMorpher
from madminer.utils.various import format_benchmark, create_missing_folders, shuffle, balance_thetas

logger = logging.getLogger(__name__)


def combine_and_shuffle(
    input_filenames, output_filename, k_factors=None, overwrite_existing_file=True, shuffle_sample=True
):
    """
    Combines multiple MadMiner files into one, and shuffles the order of the events.

    Note that this function assumes that all samples are generated with the same setup, including identical benchmarks
    (and thus morphing setup). If it is used with samples with different settings, there will be wrong results!
    There are no explicit cross checks in place yet!

    Parameters
    ----------
    input_filenames : list of str
        List of paths to the input MadMiner files.

    output_filename : str
        Path to the combined MadMiner file.

    k_factors : float or list of float, optional
        Multiplies the weights in input_filenames with a universal factor (if k_factors is a float) or with independent
        factors (if it is a list of float). Default value: None.

    overwrite_existing_file : bool, optional
        If True and if the output file exists, it is overwritten. Default value: True.

    shuffle_sample : bool, optional
        If True, the output shuffle will be shuffled. Default value: True.

    Returns
    -------
        None

    """

    logger.debug("Combining and shuffling samples")

    if len(input_filenames) > 1:
        logger.warning(
            "Careful: this tool assumes that all samples are generated with the same setup, including"
            " identical benchmarks (and thus morphing setup). If it is used with samples with different"
            " settings, there will be wrong results! There are no explicit cross checks in place yet."
        )

    # k factors
    if k_factors is None:
        k_factors = [1.0 for _ in input_filenames]
    elif isinstance(k_factors, float):
        k_factors = [k_factors for _ in input_filenames]

    # Copy first file to output_filename
    logger.info("Copying setup from %s to %s", input_filenames[0], output_filename)

    # TODO: More memory efficient strategy

    # Load events
    all_observations = None
    all_weights = None

    for i, (filename, k_factor) in enumerate(zip(input_filenames, k_factors)):
        logger.info(
            "Loading samples from file %s / %s at %s, multiplying weights with k factor %s",
            i + 1,
            len(input_filenames),
            filename,
            k_factor,
        )

        for observations, weights in madminer_event_loader(filename):
            if all_observations is None:
                all_observations = observations
                all_weights = k_factor * weights
            else:
                all_observations = np.vstack((all_observations, observations))
                all_weights = np.vstack((all_weights, k_factor * weights))

    # Shuffle
    if shuffle_sample:
        all_observations, all_weights = shuffle(all_observations, all_weights)

    # Save result
    save_preformatted_events_to_madminer_file(
        filename=output_filename,
        observations=all_observations,
        weights=all_weights,
        copy_setup_from=input_filenames[0],
        overwrite_existing_samples=overwrite_existing_file,
    )


def constant_benchmark_theta(benchmark_name):
    """
    Utility function to be used as input to various SampleAugmenter functions, specifying a single parameter benchmark.

    Parameters
    ----------
    benchmark_name : str
        Name of the benchmark (as in `madminer.core.MadMiner.add_benchmark`)
        

    Returns
    -------
    output : tuple
        Input to various SampleAugmenter functions

    """
    return "benchmark", benchmark_name


def multiple_benchmark_thetas(benchmark_names):
    """
    Utility function to be used as input to various SampleAugmenter functions, specifying multiple parameter benchmarks.

    Parameters
    ----------
    benchmark_names : list of str
        List of names of the benchmarks (as in `madminer.core.MadMiner.add_benchmark`)


    Returns
    -------
    output : tuple
        Input to various SampleAugmenter functions

    """
    return "benchmarks", benchmark_names


def constant_morphing_theta(theta):
    """
    Utility function to be used as input to various SampleAugmenter functions, specifying a single parameter point theta
    in a morphing setup.

    Parameters
    ----------
    theta : ndarray or list
        Parameter point with shape `(n_parameters,)`

    Returns
    -------
    output : tuple
        Input to various SampleAugmenter functions

    """
    return "theta", np.asarray(theta)


def multiple_morphing_thetas(thetas):
    """
    Utility function to be used as input to various SampleAugmenter functions, specifying multiple parameter points
    theta in a morphing setup.

    Parameters
    ----------
    thetas : ndarray or list of lists or list of ndarrays
        Parameter points with shape `(n_thetas, n_parameters)`

    Returns
    -------
    output : tuple
        Input to various SampleAugmenter functions

    """
    return "thetas", [np.asarray(theta) for theta in thetas]


def random_morphing_thetas(n_thetas, priors):
    """
    Utility function to be used as input to various SampleAugmenter functions, specifying random parameter points
    sampled from a prior in a morphing setup.

    Parameters
    ----------
    n_thetas : int
        Number of parameter points to be sampled

    priors : list of tuples
        Priors for each parameter is characterized by a tuple of the form `(prior_shape, prior_param_0, prior_param_1)`.
        Currently, the supported prior_shapes are `flat`, in which case the two other parameters are the lower and upper
        bound of the flat prior, and `gaussian`, in which case they are the mean and standard deviation of a Gaussian.

    Returns
    -------
    output : tuple
        Input to various SampleAugmenter functions

    """
    return "random", (n_thetas, priors)


class SampleAugmenter:
    """
    Sampling and data augmentation.

    After the generated events have been analyzed and the observables and weights have been saved into a MadMiner file,
    for instance with `madminer.delphes.DelphesProcessor` or `madminer.lhe.LHEProcessor`, the next step is typically
    the generation of training and evaluation data for the machine learning algorithms. This generally involves two
    (related) tasks: unweighting, i.e. the creation of samples that do not carry individual weights but follow some
    distribution, and the extraction of the joint likelihood ratio and / or joint score (the "augmented data").

    After inializing `SampleAugmenter` with the filename of a MadMiner file, this is done with a single function call.
    Depending on the downstream inference algorithm, there are different possibilities:

    * `SampleAugmenter.extract_samples_train_plain()` creates plain training samples without augmented data.
    * `SampleAugmenter.extract_samples_train_local()` creates training samples for local methods based on the score,
      such as SALLY and SALLINO.
    * `SampleAugmenter.extract_samples_train_global()` creates training samples for non-local methods based on density
      estimation and the score, such as SCANDAL.
    * `SampleAugmenter.extract_samples_train_ratio()` creates training samples for non-local, ratio-based methods
      like RASCAL or ALICE.
    * `SampleAugmenter.extract_samples_train_more_ratios()` does the same, but can extract joint ratios and scores
      at more parameter points. This additional information  can be used efficiently in the setup with a "doubly
      parameterized" likelihood ratio estimator that models the dependence on both the numerator and denominator
      hypothesis.
    * `SampleAugmenter.extract_samples_test()` creates evaluation samples for all methods.

    Please see the tutorial for a walkthrough.

    For the curious, let us explain these steps in a little bit more detail (assuming a morphing setup):

    * The sample augmentation step starts from a set of events `(x_i, z_i)` together with corresponding weights for each
      morphing basis point `theta_b`, `p(x_i, z_i | theta_b)`.
    * Morphing: Assume we want to generate data sampled from a parameter point theta, which is not necessarily one of
      the basis points theta_b. Using the morphing structure, the event weights for p(x_i, z_i | theta) can be
      calculated. Note that the events (phase-space points) `(x_i, z_i)` are not changed, only their weights.
    * Unweighting: For the machine learning part, such a weighted event sample is not practical. Instead we aim for an
      unweighted one, in which events can appear multiple times. If the user request `N` events (which can be larger
      than the original number of events in the MadGraph runs), SampleAugmenter will draw `N` samples `(x_i, z_i)` from
      the discrete distribution `p(x_i, z_i | theta)`. In other words, it draws (with replacement) `N` of the original
      events from MadGraph, with probabilities given by the morphing setup before. This is similar to what
      `np.random.choice()` does.
    * Augmentation: For each of the drawn samples, the morphing setup can be used to calculate the joint likelihood
      ratio and / or the joint score (this depends on which SampleAugmenter function is called).

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
            self.morpher = Morpher(self.parameters)
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

    def extract_samples_train_plain(
        self, theta, n_samples, folder, filename, test_split=0.5, switch_train_test_events=False
    ):
        """
        Extracts plain training samples `x ~ p(x|theta)` without any augmented data. This can be use for standard
        inference methods such as ABC, histograms of observables, or neural density estimation techniques. It can also
        be used to create validation or calibration samples.

        Parameters
        ----------
        theta : tuple
            Tuple (type, value) that defines the parameter point or prior over parameter points for the
            sampling. Pass the output of the functions `constant_benchmark_theta()`, `multiple_benchmark_thetas()`,
            `constant_morphing_theta()`, `multiple_morphing_thetas()`, or `random_morphing_thetas()`.

        n_samples : int
            Total number of events to be drawn.

        folder : str
            Path to the folder where the resulting samples should be saved (ndarrays in .npy format).

        filename : str
            Filenames for the resulting samples. A prefix such as 'x' or 'theta0' as well as the extension
            '.npy' will be added automatically.

        test_split : float or None, optional
            Fraction of events reserved for the evaluation sample (that will not be used for any training samples).
            Default value: 0.5.

        switch_train_test_events : bool, optional
            If True, this function generates a training sample from the events normally reserved for test samples.
            Default value: False.

        Returns
        -------
        x : ndarray
            Observables with shape `(n_samples, n_observables)`. The same information is saved as a file in the given
            folder.

        theta : ndarray
            Parameter points used for sampling with shape `(n_samples, n_parameters)`. The same information is saved as
            a file in the given folder.

        """

        logger.info("Extracting plain training sample. Sampling according to %s", theta)

        create_missing_folders([folder])

        # Thetas
        theta_types, theta_values, n_samples_per_theta = parse_theta(theta, n_samples)

        # Train / test split
        start_event, end_event = self._train_test_split(not switch_train_test_events, test_split)

        # Start
        x, _, (theta,) = self._extract_sample(
            theta_sets_types=[theta_types],
            theta_sets_values=[theta_values],
            n_samples_per_theta=n_samples_per_theta,
            start_event=start_event,
            end_event=end_event,
        )

        # Save data
        if filename is not None and folder is not None:
            np.save(folder + "/theta_" + filename + ".npy", theta)
            np.save(folder + "/x_" + filename + ".npy", x)

        return x, theta

    def extract_samples_train_local(
        self,
        theta,
        n_samples,
        folder,
        filename,
        nuisance_score=False,
        test_split=0.5,
        switch_train_test_events=False,
        log_message=True,
    ):
        """
        Extracts training samples x ~ p(x|theta) as well as the joint score t(x, z|theta). This can be used for
        inference methods such as SALLY and SALLINO.

        Parameters
        ----------
        theta : tuple
            Tuple (type, value) that defines the parameter point for the sampling. This is also where the score is
            evaluated. Pass the output of the functions `constant_benchmark_theta()` or `constant_morphing_theta()`.

        n_samples : int
            Total number of events to be drawn.

        folder : str
            Path to the folder where the resulting samples should be saved (ndarrays in .npy format).

        filename : str
            Filenames for the resulting samples. A prefix such as 'x' or 'theta0' as well as the extension
            '.npy' will be added automatically.

        nuisance_score : bool, optional
            If True and if the sample contains nuisance parameters, the score with respect to the nuisance parameters
            (at the default position) will also be calculated. Otherwise, only the score with respect to the
            physics parameters is calculated. Default: False.

        test_split : float or None, optional
            Fraction of events reserved for the evaluation sample (that will not be used for any training samples).
            Default value: 0.5.

        switch_train_test_events : bool, optional
            If True, this function generates a training sample from the events normally reserved for test samples.
            Default value: False.

        log_message : bool, optional
            If True, logging output. This option is only designed for internal use.

        Returns
        -------
        x : ndarray
            Observables with shape `(n_samples, n_observables)`. The same information is saved as a file in the given
            folder.

        theta : ndarray
            Parameter points used for sampling (and  evaluation of the joint score) with shape
            `(n_samples, n_parameters)`. The same information is saved as a file in the given folder.

        t_xz : ndarray
            Joint score evaluated at theta with shape `(n_samples, n_parameters + n_nuisance_parameters)` (if
            nuisance_score is True) or `(n_samples, n_parameters)`. The same information is saved as a
            file in the given folder.

        """

        if log_message:
            logger.info(
                "Extracting training sample for local score regression. Sampling and score evaluation according to %s",
                theta,
            )

        create_missing_folders([folder])

        # Check setup
        if self.morpher is None:
            raise RuntimeError("No morphing setup loaded. Cannot calculate score.")

        if self.nuisance_morpher is None and nuisance_score:
            raise RuntimeError("No nuisance parameters defined. Cannot calculate nuisance score.")

        # Thetas
        theta_types, theta_values, n_samples_per_theta = parse_theta(theta, n_samples)

        # Augmented data (gold)
        augmented_data_definitions = [("score", 0)]
        if nuisance_score:
            augmented_data_definitions += [("nuisance_score",)]

        # Train / test split
        start_event, end_event = self._train_test_split(not switch_train_test_events, test_split)

        # Start
        x, augmented_data, (theta,) = self._extract_sample(
            theta_sets_types=[theta_types],
            theta_sets_values=[theta_values],
            n_samples_per_theta=n_samples_per_theta,
            augmented_data_definitions=augmented_data_definitions,
            nuisance_score=nuisance_score,
            start_event=start_event,
            end_event=end_event,
        )

        t_xz_physics = augmented_data[0]
        if nuisance_score:
            t_xz_nuisance = augmented_data[1]
            t_xz = np.hstack([t_xz_physics, t_xz_nuisance])

            logger.debug(
                "Found physical score with shape %s, nuisance score with shape %s, combined shape %s",
                t_xz_physics.shape,
                t_xz_nuisance.shape,
                t_xz.shape,
            )
        else:
            t_xz = t_xz_physics

        # Save data
        if filename is not None and folder is not None:
            np.save(folder + "/theta_" + filename + ".npy", theta)
            np.save(folder + "/x_" + filename + ".npy", x)
            np.save(folder + "/t_xz_" + filename + ".npy", t_xz)

        return x, theta, t_xz

    def extract_samples_train_global(
        self, theta, n_samples, folder, filename, test_split=0.5, switch_train_test_events=False
    ):
        """
        Extracts training samples x ~ p(x|theta) as well as the joint score t(x, z|theta), where theta is sampled
        from a prior. This can be used for inference methods such as SCANDAL.

        Parameters
        ----------
        theta : tuple
            Tuple (type, value) that defines the numerator parameter point or prior over parameter points for the
            sampling. Pass the output of the functions `constant_benchmark_theta()`, `multiple_benchmark_thetas()`,
            `constant_morphing_theta()`, `multiple_morphing_thetas()`, or `random_morphing_thetas()`.

        n_samples : int
            Total number of events to be drawn.

        folder : str
            Path to the folder where the resulting samples should be saved (ndarrays in .npy format).

        filename : str
            Filenames for the resulting samples. A prefix such as 'x' or 'theta0' as well as the extension
            '.npy' will be added automatically.

        test_split : float or None, optional
            Fraction of events reserved for the evaluation sample (that will not be used for any training samples).
            Default value: 0.5.

        switch_train_test_events : bool, optional
            If True, this function generates a training sample from the events normally reserved for test samples.
            Default value: False.

        Returns
        -------
        x : ndarray
            Observables with shape `(n_samples, n_observables)`. The same information is saved as a file in the given
            folder.

        theta : ndarray
            Parameter points used for sampling (and  evaluation of the joint score) with shape
            `(n_samples, n_parameters)`. The same information is saved as a file in the given folder.

        t_xz : ndarray
            Joint score evaluated at theta with shape `(n_samples, n_parameters)`. The same information is saved as a
            file in the given folder.

        """

        logger.info(
            "Extracting training sample for non-local score-based methods. Sampling and score evaluation according "
            "to %s",
            theta,
        )

        return self.extract_samples_train_local(
            theta,
            n_samples,
            folder,
            filename,
            test_split=test_split,
            switch_train_test_events=switch_train_test_events,
            log_message=False,
        )

    def extract_samples_train_ratio(
        self, theta0, theta1, n_samples, folder, filename, test_split=0.5, switch_train_test_events=False
    ):
        """
        Extracts training samples `x ~ p(x|theta0)` and `x ~ p(x|theta1)` together with the class label `y`, the joint
        likelihood ratio `r(x,z|theta0, theta1)`, and, if morphing is set up, the joint score `t(x,z|theta0)`. This
        information can be used in inference methods such as CARL, ROLR, CASCAL, and RASCAL.

        Parameters
        ----------
        theta0 : tuple
            Tuple (type, value) that defines the numerator parameter point or prior over parameter points for the
            sampling. Pass the output of the functions `constant_benchmark_theta()`, `multiple_benchmark_thetas()`,
            `constant_morphing_theta()`, `multiple_morphing_thetas()`, or `random_morphing_thetas()`.

        theta1 : tuple
            Tuple (type, value) that defines the denominator parameter point or prior over parameter points for the
            sampling. Pass the output of the functions `constant_benchmark_theta()`, `multiple_benchmark_thetas()`,
            `constant_morphing_theta()`, `multiple_morphing_thetas()`, or `random_morphing_thetas()`.

        n_samples : int
            Total number of events to be drawn.

        folder : str
            Path to the folder where the resulting samples should be saved (ndarrays in .npy format).

        filename : str
            Filenames for the resulting samples. A prefix such as 'x' or 'theta0' as well as the extension
            '.npy' will be added automatically.

        test_split : float or None, optional
            Fraction of events reserved for the evaluation sample (that will not be used for any training samples).
            Default value: 0.5.

        switch_train_test_events : bool, optional
            If True, this function generates a training sample from the events normally reserved for test samples.
            Default value: False.

        Returns
        -------
        x : ndarray
            Observables with shape `(n_samples, n_observables)`. The same information is saved as a file in the given
            folder.

        theta0 : ndarray
            Numerator parameter points with shape `(n_samples, n_parameters)`. The same information is saved as
            a file in the given folder.

        theta1 : ndarray
            Denominator parameter points with shape `(n_samples, n_parameters)`. The same information is saved as
            a file in the given folder.

        y : ndarray
            Class label with shape `(n_samples, n_parameters)`. `y=0` (`1`) for events sample from the numerator
            (denominator) hypothesis. The same information is saved as a file in the given folder.

        r_xz : ndarray
            Joint likelihood ratio with shape `(n_samples,)`. The same information is saved as a file in the given
            folder.

        t_xz : ndarray or None
            If morphing is set up, the joint score evaluated at theta0 with shape `(n_samples, n_parameters)`. The same
            information is saved as a file in the given folder. If morphing is not set up, None is returned (and no
            file is saved).

        """

        logger.info(
            "Extracting training sample for ratio-based methods. Numerator hypothesis: %s, denominator "
            "hypothesis: %s",
            theta0,
            theta1,
        )

        if self.morpher is None:
            logging.warning("No morphing setup loaded. Cannot calculate joint score.")

        create_missing_folders([folder])

        # Augmented data (gold)
        augmented_data_definitions = [("ratio", 0, 1)]
        if self.morpher is not None:
            augmented_data_definitions.append(("score", 0))

        # Train / test split
        start_event, end_event = self._train_test_split(not switch_train_test_events, test_split)

        # Thetas for theta0 sampling
        theta0_types, theta0_values, n_samples_per_theta0 = parse_theta(theta0, n_samples // 2)
        theta1_types, theta1_values, n_samples_per_theta1 = parse_theta(theta1, n_samples // 2)

        n_samples_per_theta = min(n_samples_per_theta0, n_samples_per_theta1)

        # Start for theta0

        if self.morpher is None:
            x0, (r_xz0,), (theta0_0, theta1_0) = self._extract_sample(
                theta_sets_types=[theta0_types, theta1_types],
                theta_sets_values=[theta0_values, theta1_values],
                sampling_theta_index=0,
                n_samples_per_theta=n_samples_per_theta,
                augmented_data_definitions=augmented_data_definitions,
                start_event=start_event,
                end_event=end_event,
            )
            t_xz0 = None
        else:
            x0, (r_xz0, t_xz0), (theta0_0, theta1_0) = self._extract_sample(
                theta_sets_types=[theta0_types, theta1_types],
                theta_sets_values=[theta0_values, theta1_values],
                sampling_theta_index=0,
                n_samples_per_theta=n_samples_per_theta,
                augmented_data_definitions=augmented_data_definitions,
                start_event=start_event,
                end_event=end_event,
            )

        # Thetas for theta1 sampling (could be different if num or denom are random)
        theta0_types, theta0_values, n_samples_per_theta0 = parse_theta(theta0, n_samples // 2)
        theta1_types, theta1_values, n_samples_per_theta1 = parse_theta(theta1, n_samples // 2)

        n_samples_per_theta = min(n_samples_per_theta0, n_samples_per_theta1)

        # Start for theta1
        if self.morpher is None:
            x1, (r_xz1,), (theta0_1, theta1_1) = self._extract_sample(
                theta_sets_types=[theta0_types, theta1_types],
                theta_sets_values=[theta0_values, theta1_values],
                sampling_theta_index=1,
                n_samples_per_theta=n_samples_per_theta,
                augmented_data_definitions=augmented_data_definitions,
                start_event=start_event,
                end_event=end_event,
            )
            t_xz1 = None
        else:
            x1, (r_xz1, t_xz1), (theta0_1, theta1_1) = self._extract_sample(
                theta_sets_types=[theta0_types, theta1_types],
                theta_sets_values=[theta0_values, theta1_values],
                sampling_theta_index=1,
                n_samples_per_theta=n_samples_per_theta,
                augmented_data_definitions=augmented_data_definitions,
                start_event=start_event,
                end_event=end_event,
            )

        # Combine
        x = np.vstack([x0, x1])
        r_xz = np.vstack([r_xz0, r_xz1])
        if self.morpher is not None:
            t_xz = np.vstack([t_xz0, t_xz1])
        else:
            t_xz = None
        theta0 = np.vstack([theta0_0, theta0_1])
        theta1 = np.vstack([theta1_0, theta1_1])
        y = np.zeros(x.shape[0])
        y[x0.shape[0] :] = 1.0

        # Shuffle
        x, r_xz, t_xz, theta0, theta1, y = shuffle(x, r_xz, t_xz, theta0, theta1, y)

        # y shape
        y = y.reshape((-1, 1))

        # Save data
        if filename is not None and folder is not None:
            np.save(folder + "/theta0_" + filename + ".npy", theta0)
            np.save(folder + "/theta1_" + filename + ".npy", theta1)
            np.save(folder + "/x_" + filename + ".npy", x)
            np.save(folder + "/y_" + filename + ".npy", y)
            np.save(folder + "/r_xz_" + filename + ".npy", r_xz)
            if self.morpher is not None:
                np.save(folder + "/t_xz_" + filename + ".npy", t_xz)

        return x, theta0, theta1, y, r_xz, t_xz

    def extract_samples_train_more_ratios(
        self,
        theta0,
        theta1,
        n_samples,
        folder,
        filename,
        additional_thetas=None,
        test_split=0.5,
        switch_train_test_events=False,
    ):
        """
        Extracts training samples `x ~ p(x|theta0)` and `x ~ p(x|theta1)` together with the class label `y`, the joint
        likelihood ratio `r(x,z|theta0, theta1)`, and the joint score `t(x,z|theta0)`. This information can be used in
        inference methods such as CARL, ROLR, CASCAL, and RASCAL.

        With the keyword `additional_thetas`, this function allows to extract joint ratios and scores
        at more parameter points than just `theta0` and `theta1`. This additional information can be used efficiently
        in the setup with a "doubly parameterized" likelihood ratio estimator that models the dependence on both the
        numerator and denominator hypothesis.

        Parameters
        ----------
        theta0 :
            Tuple (type, value) that defines the numerator parameter point or prior over parameter points for the
            sampling. Pass the output of the functions `constant_benchmark_theta()`, `multiple_benchmark_thetas()`,
            `constant_morphing_theta()`, `multiple_morphing_thetas()`, or `random_morphing_thetas()`.

        theta1 :
            Tuple (type, value) that defines the denominator parameter point or prior over parameter points for the
            sampling. Pass the output of the functions `constant_benchmark_theta()`, `multiple_benchmark_thetas()`,
            `constant_morphing_theta()`, `multiple_morphing_thetas()`, or `random_morphing_thetas()`.

        n_samples : int
            Total number of events to be drawn.

        folder : str
            Path to the folder where the resulting samples should be saved (ndarrays in .npy format).

        filename : str
            Filenames for the resulting samples. A prefix such as 'x' or 'theta0' as well as the extension
            '.npy' will be added automatically.

        additional_thetas : list of tuple or None
            list of tuples `(type, value)` that defines additional theta points at which ratio and score are evaluated,
            and which are then used to create additional training data points. These can be efficiently used only in
            the "doubly parameterized" setup where a likelihood ratio estimator models the dependence of the likelihood
            ratio on both the numerator and denominator hypothesis. Pass the output of  the helper functions
            `constant_benchmark_theta()`, `multiple_benchmark_thetas()`, `constant_morphing_theta()`,
            `multiple_morphing_thetas()`, or `random_morphing_thetas()`. Default value: None.

        test_split : float or None, optional
            Fraction of events reserved for the evaluation sample (that will not be used for any training samples).
            Default value: 0.5.

        switch_train_test_events : bool, optional
            If True, this function generates a training sample from the events normally reserved for test samples.
            Default value: False.

        Returns
        -------
        x : ndarray
            Observables with shape `(n_samples, n_observables)`. The same information is saved as a file in the given
            folder.

        theta0 : ndarray
            Numerator parameter points with shape `(n_samples, n_parameters)`. The same information is saved as
            a file in the given folder.

        theta1 : ndarray
            Denominator parameter points with shape `(n_samples, n_parameters)`. The same information is saved as
            a file in the given folder.

        y : ndarray
            Class label with shape `(n_samples, n_parameters)`. `y=0` (`1`) for events sample from the numerator
            (denominator) hypothesis. The same information is saved as a file in the given folder.

        r_xz : ndarray
            Joint likelihood ratio with shape `(n_samples,)`. The same information is saved as a file in the given
            folder.

        t_xz : ndarray
            Joint score evaluated at theta0 with shape `(n_samples, n_parameters)`. The same information is saved as a
            file in the given folder.

        """

        logger.info(
            "Extracting training sample for ratio-based methods. Numerator hypothesis: %s, denominator "
            "hypothesis: %s",
            theta0,
            theta1,
        )

        if self.morpher is None:
            raise RuntimeError("No morphing setup loaded. Cannot calculate score.")

        create_missing_folders([folder])

        if additional_thetas is None:
            additional_thetas = []
        n_additional_thetas = len(additional_thetas)

        # Augmented data (gold)
        augmented_data_definitions_0 = [("ratio", 0, 1), ("score", 0), ("score", 1)]
        augmented_data_definitions_1 = [("ratio", 0, 1), ("score", 0), ("score", 1)]
        for i in range(n_additional_thetas):
            augmented_data_definitions_0.append(("ratio", 0, i + 2))
            augmented_data_definitions_0.append(("score", i + 2))
            augmented_data_definitions_1.append(("ratio", i + 2, 1))
            augmented_data_definitions_1.append(("score", i + 2))

        # Train / test split
        start_event, end_event = self._train_test_split(not switch_train_test_events, test_split)

        # Parse thetas for theta0 sampling
        theta_types = []
        theta_values = []
        n_samples_per_theta = 1000000

        theta0_types, theta0_values, this_n_samples = parse_theta(theta0, n_samples // 2)
        theta_types.append(theta0_types)
        theta_values.append(theta0_values)
        n_samples_per_theta = min(this_n_samples, n_samples_per_theta)

        theta1_types, theta1_values, this_n_samples = parse_theta(theta1, n_samples // 2)
        theta_types.append(theta1_types)
        theta_values.append(theta1_values)
        n_samples_per_theta = min(this_n_samples, n_samples_per_theta)

        for additional_theta in additional_thetas:
            additional_theta_types, additional_theta_values, this_n_samples = parse_theta(
                additional_theta, n_samples // 2
            )
            theta_types.append(additional_theta_types)
            theta_values.append(additional_theta_values)
            n_samples_per_theta = min(this_n_samples, n_samples_per_theta)

        # Start for theta0
        x_0, augmented_data_0, thetas_0 = self._extract_sample(
            theta_sets_types=theta_types,
            theta_sets_values=theta_values,
            n_samples_per_theta=n_samples_per_theta,
            augmented_data_definitions=augmented_data_definitions_0,
            sampling_theta_index=0,
            start_event=start_event,
            end_event=end_event,
        )
        n_actual_samples = x_0.shape[0]

        # Analyse theta values from theta0 run
        theta0_0 = thetas_0[0]
        theta1_0 = thetas_0[1]
        thetas_eval = thetas_0[2:]

        # Analyse augmented data from theta0 run
        r_xz_0 = augmented_data_0[0]
        t_xz0_0 = augmented_data_0[1]
        t_xz1_0 = augmented_data_0[2]

        r_xz_eval = []
        t_xz_eval = []
        for i, theta_eval in enumerate(thetas_eval):
            r_xz_eval.append(augmented_data_0[3 + i * 2])
            t_xz_eval.append(augmented_data_0[4 + i * 2])

        x_0 = np.vstack([x_0 for _ in range(1 + n_additional_thetas)])
        r_xz_0 = np.vstack([r_xz_0] + r_xz_eval)
        t_xz0_0 = np.vstack([t_xz0_0 for _ in range(1 + n_additional_thetas)])
        t_xz1_0 = np.vstack([t_xz1_0] + t_xz_eval)
        theta0_0 = np.vstack([theta0_0 for _ in range(1 + n_additional_thetas)])
        theta1_0 = np.vstack([theta1_0] + thetas_eval)

        # Parse thetas for theta1 sampling
        theta_types = []
        theta_values = []
        n_samples_per_theta = 1000000

        theta0_types, theta0_values, this_n_samples = parse_theta(theta0, n_samples // 2)
        theta_types.append(theta0_types)
        theta_values.append(theta0_values)
        n_samples_per_theta = min(this_n_samples, n_samples_per_theta)

        theta1_types, theta1_values, this_n_samples = parse_theta(theta1, n_samples // 2)
        theta_types.append(theta1_types)
        theta_values.append(theta1_values)
        n_samples_per_theta = min(this_n_samples, n_samples_per_theta)

        for additional_theta in additional_thetas:
            additional_theta_types, additional_theta_values, this_n_samples = parse_theta(
                additional_theta, n_samples // 2
            )
            theta_types.append(additional_theta_types)
            theta_values.append(additional_theta_values)
            n_samples_per_theta = min(this_n_samples, n_samples_per_theta)

        # Start for theta1
        x_1, augmented_data_1, thetas_1 = self._extract_sample(
            theta_sets_types=theta_types,
            theta_sets_values=theta_values,
            n_samples_per_theta=n_samples_per_theta,
            augmented_data_definitions=augmented_data_definitions_1,
            sampling_theta_index=1,
            start_event=start_event,
            end_event=end_event,
        )
        n_actual_samples += x_1.shape[0]

        # Analyse theta values from theta1 run
        theta0_1 = thetas_1[0]
        theta1_1 = thetas_1[1]
        thetas_eval = thetas_1[2:]

        # Analyse augmented data from theta1 run
        r_xz_1 = augmented_data_1[0]
        t_xz0_1 = augmented_data_1[1]
        t_xz1_1 = augmented_data_1[2]

        r_xz_eval = []
        t_xz_eval = []
        for i, theta_eval in enumerate(thetas_eval):
            r_xz_eval.append(augmented_data_1[3 + i * 2])
            t_xz_eval.append(augmented_data_1[4 + i * 2])

        x_1 = np.vstack([x_1 for _ in range(1 + n_additional_thetas)])
        r_xz_1 = np.vstack([r_xz_1] + r_xz_eval)
        t_xz0_1 = np.vstack([t_xz0_1] + t_xz_eval)
        t_xz1_1 = np.vstack([t_xz1_1 for _ in range(1 + n_additional_thetas)])
        theta0_1 = np.vstack([theta0_1] + thetas_eval)
        theta1_1 = np.vstack([theta1_1 for _ in range(1 + n_additional_thetas)])

        # Combine
        x = np.vstack([x_0, x_1])
        r_xz = np.vstack([r_xz_0, r_xz_1])
        t_xz0 = np.vstack([t_xz0_0, t_xz0_1])
        t_xz1 = np.vstack([t_xz1_0, t_xz1_1])
        theta0 = np.vstack([theta0_0, theta0_1])
        theta1 = np.vstack([theta1_0, theta1_1])
        y = np.zeros(x.shape[0])
        y[x_0.shape[0] :] = 1.0

        if n_additional_thetas > 0:
            logger.info(
                "Oversampling: created %s training samples from %s original unweighted events",
                x.shape[0],
                n_actual_samples,
            )

        # Shuffle
        x, r_xz, t_xz0, t_xz1, theta0, theta1, y = shuffle(x, r_xz, t_xz0, t_xz1, theta0, theta1, y)

        # y shape
        y = y.reshape((-1, 1))

        # Save data
        if filename is not None and folder is not None:
            np.save(folder + "/theta0_" + filename + ".npy", theta0)
            np.save(folder + "/theta1_" + filename + ".npy", theta1)
            np.save(folder + "/x_" + filename + ".npy", x)
            np.save(folder + "/y_" + filename + ".npy", y)
            np.save(folder + "/r_xz_" + filename + ".npy", r_xz)
            np.save(folder + "/t_xz0_" + filename + ".npy", t_xz0)
            np.save(folder + "/t_xz1_" + filename + ".npy", t_xz1)

        return x, theta0, theta1, y, r_xz, t_xz0, t_xz1

    def extract_samples_test(self, theta, n_samples, folder, filename, test_split=0.5, switch_train_test_events=False):
        """
        Extracts evaluation samples `x ~ p(x|theta)` without any augmented data.

        Parameters
        ----------
        theta : tuple
            Tuple (type, value) that defines the parameter point or prior over parameter points for the
            sampling. Pass the output of the functions `constant_benchmark_theta()`, `multiple_benchmark_thetas()`,
            `constant_morphing_theta()`, `multiple_morphing_thetas()`, or `random_morphing_thetas()`.

        n_samples : int
            Total number of events to be drawn.

        folder : str
            Path to the folder where the resulting samples should be saved (ndarrays in .npy format).

        filename : str
            Filenames for the resulting samples. A prefix such as 'x' or 'theta0' as well as the extension
            '.npy' will be added automatically.

        test_split : float or None, optional
            Fraction of events reserved for the evaluation sample (that will not be used for any training samples).
            Default value: 0.5.

        switch_train_test_events : bool, optional
            If True, this function generates a test sample from the events normally reserved for training samples.
            Default value: False.

        Returns
        -------
        x : ndarray
            Observables with shape `(n_samples, n_observables)`. The same information is saved as a file in the given
            folder.

        theta : ndarray
            Parameter points used for sampling with shape `(n_samples, n_parameters)`. The same information is saved as
            a file in the given folder.

        """

        logger.info("Extracting evaluation sample. Sampling according to %s", theta)

        create_missing_folders([folder])

        # Thetas
        theta_types, theta_values, n_samples_per_theta = parse_theta(theta, n_samples)

        # Train / test split
        start_event, end_event = self._train_test_split(switch_train_test_events, test_split)

        # Extract information
        x, _, (theta,) = self._extract_sample(
            theta_sets_types=[theta_types],
            theta_sets_values=[theta_values],
            n_samples_per_theta=n_samples_per_theta,
            start_event=start_event,
            end_event=end_event,
        )

        # Save data
        if filename is not None and folder is not None:
            np.save(folder + "/theta_" + filename + ".npy", theta)
            np.save(folder + "/x_" + filename + ".npy", x)

        return x, theta

    def extract_cross_sections(self, theta):

        """
        Calculates the total cross sections for all specified thetas.

        Parameters
        ----------
        theta : tuple
            Tuple (type, value) that defines the parameter point or prior over parameter points at which the cross
            section is calculated. Pass the output of the functions `constant_benchmark_theta()`,
            `multiple_benchmark_thetas()`, `constant_morphing_theta()`, `multiple_morphing_thetas()`, or
            `random_morphing_thetas()`.

        Returns
        -------
        thetas : ndarray
            Parameter points with shape `(n_thetas, n_parameters)`.

        xsecs : ndarray
            Total cross sections in pb with shape `(n_thetas, )`.

        xsec_uncertainties : ndarray
            Statistical uncertainties on the total cross sections in pb with shape `(n_thetas, )`.

        """

        logger.info("Starting cross-section calculation")

        # Total xsecs for benchmarks
        xsecs_benchmarks = None
        squared_weight_sum_benchmarks = None

        for obs, weights in madminer_event_loader(self.madminer_filename):
            if xsecs_benchmarks is None:
                xsecs_benchmarks = np.sum(weights, axis=0)
                squared_weight_sum_benchmarks = np.sum(weights * weights, axis=0)
            else:
                xsecs_benchmarks += np.sum(weights, axis=0)
                squared_weight_sum_benchmarks += np.sum(weights * weights, axis=0)

        # Parse thetas for evaluation
        theta_types, theta_values, _ = parse_theta(theta, 1)

        # Loop over thetas
        all_thetas = []
        all_xsecs = []
        all_xsec_uncertainties = []

        for (theta_type, theta_value) in zip(theta_types, theta_values):

            if self.morpher is None and theta_type == "morphing":
                raise RuntimeError("Theta defined through morphing, but no morphing setup has been loaded.")

            theta = get_theta_value(theta_type, theta_value, self.benchmarks)
            theta_matrix = get_theta_benchmark_matrix(theta_type, theta_value, self.benchmarks, self.morpher)

            # Total xsec for this theta
            xsec_theta = mdot(theta_matrix, xsecs_benchmarks)
            rms_xsec_theta = mdot(theta_matrix * theta_matrix, squared_weight_sum_benchmarks) ** 0.5

            all_thetas.append(theta)
            all_xsecs.append(xsec_theta)
            all_xsec_uncertainties.append(rms_xsec_theta)

            logger.debug("theta %s: xsec = (%s +/- %s) pb", theta, xsec_theta, rms_xsec_theta)

        # Return
        all_thetas = np.array(all_thetas)
        all_xsecs = np.array(all_xsecs)
        all_xsec_uncertainties = np.array(all_xsec_uncertainties)

        return all_thetas, all_xsecs, all_xsec_uncertainties

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

    def _extract_sample(
        self,
        theta_sets_types,
        theta_sets_values,
        n_samples_per_theta,
        sampling_theta_index=0,
        augmented_data_definitions=None,
        nuisance_score=False,
        start_event=0,
        end_event=None,
    ):
        """
        Low-level function for the extraction of information from the event samples. Do not use this function directly.

        Parameters
        ----------
        theta_sets_types :  list of list of str
            Each entry can be 'benchmark' or 'morphing'.

        theta_sets_values : list of list
            Each entry is int and labels the benchmark index (if the corresponding
            theta_sampling_types entry is 'benchmark') or a numpy array with the theta values
            (of the corresponding theta_sampling_types entry is 'morphing')

        n_samples_per_theta : int
            Number of samples to be drawn per entry in theta_sampling_types.

        augmented_data_definitions : list of tuple or None
            Each tuple can either be ('ratio', num_theta, den_theta) or
            ('score', theta), where num_theta, den_theta, and theta are indexes marking
            which of the theta sets defined through thetas_types and thetas_values is
            used. Default value: None.

        nuisance_score : bool, optional
            If True and if the sample contains nuisance parameters, any joint score in the augmented data definitions
            is also calculated with respect to the nuisance parameters (evaluated at their default position). Default
            value: False.

        sampling_theta_index : int
            Marking the index of the theta set defined through thetas_types and
            thetas_values that should be used for sampling. Default value: 0.

        start_event : int
            Index of first event to consider. Default value: 0.

        end_event : int or None
            Index of last event to consider. If None, use the last event. Default value: None.

        Returns
        -------
        x :  ndarray
            Observables.

        augmented_data : list of ndarray
            Augmented data.

        theta : list of ndarray
            Parameter values.

        """

        logger.debug("Starting sample extraction")

        assert n_samples_per_theta > 0, "Requested {} samples per theta!".format(n_samples_per_theta)

        if augmented_data_definitions is None:
            augmented_data_definitions = []

        logger.debug("Augmented data requested:")
        for augmented_data_definition in augmented_data_definitions:
            logger.debug("  %s", augmented_data_definition)

        # Nuisance parameters?
        include_nuisance_parameters = self.include_nuisance_parameters and nuisance_score

        # Calculate total xsecs for benchmarks
        xsecs_benchmarks = None
        squared_weight_sum_benchmarks = None
        n_observables = 0

        for obs, weights in madminer_event_loader(
            self.madminer_filename,
            start=start_event,
            end=end_event,
            include_nuisance_parameters=include_nuisance_parameters,
            benchmark_is_nuisance=self.benchmark_is_nuisance,
        ):
            # obs has shape (n_events, n_observables)
            # weights has shape (n_events, n_benchmarks_phys)
            # sampled_from_benchmark has shape (n_events,)

            if xsecs_benchmarks is None:
                xsecs_benchmarks = np.sum(weights, axis=0)
                squared_weight_sum_benchmarks = np.sum(weights * weights, axis=0)
            else:
                xsecs_benchmarks += np.sum(weights, axis=0)
                squared_weight_sum_benchmarks += np.sum(weights * weights, axis=0)

            n_observables = obs.shape[1]

        logger.debug("Benchmark cross sections [pb]: %s", xsecs_benchmarks)

        # Balance thetas
        theta_sets_types, theta_sets_values = balance_thetas(theta_sets_types, theta_sets_values)

        # Check whether we need to calculate scores (which will require the gradients of the morphing matrices)
        needs_gradients = False
        for augmented_data_definition in augmented_data_definitions:
            if augmented_data_definition[0] == "score":
                needs_gradients = True

                if self.morpher is None:
                    raise RuntimeError("Cannot calculate score without morphing setup!")

        # Consistency checks
        n_benchmarks = xsecs_benchmarks.shape[0]
        expected_n_benchmarks = self.n_benchmarks if include_nuisance_parameters else self.n_benchmarks_phys
        if self.morphing_matrix is None:
            if n_benchmarks != expected_n_benchmarks:
                raise ValueError(
                    "Inconsistent numbers of benchmarks: {} in observations,"
                    "{} in benchmark list".format(n_benchmarks, len(self.benchmarks))
                )
        else:
            if n_benchmarks != expected_n_benchmarks or n_benchmarks < self.morphing_matrix.shape[0]:
                raise ValueError(
                    "Inconsistent numbers of benchmarks: {} in observations, {} in benchmark list, "
                    "{} in morphing matrix".format(n_benchmarks, len(self.benchmarks), self.morphing_matrix.shape[0])
                )

        if n_observables != len(self.observables):
            raise ValueError(
                "Inconsistent numbers of observables: {} in observations,"
                "{} in observable list".format(n_observables, len(self.observables))
            )

        n_thetas = len(theta_sets_types)
        assert n_thetas == len(theta_sets_values)
        # Sets (within each set, all thetas (sampling, numerator, ...) have a constant value)
        n_sets = len(theta_sets_types[sampling_theta_index])
        for theta_types, theta_values in zip(theta_sets_types, theta_sets_values):
            assert n_sets == len(theta_types) == len(theta_values)

        # Number of samples to be drawn
        if not isinstance(n_samples_per_theta, collections.Iterable):
            n_samples_per_theta = [n_samples_per_theta] * n_sets
        elif len(n_samples_per_theta) == 1:
            n_samples_per_theta = [n_samples_per_theta[0]] * n_sets

        # Prepare output
        all_x = []
        all_augmented_data = [[] for _ in augmented_data_definitions]
        all_thetas = [[] for _ in range(n_thetas)]
        all_effective_n_samples = []

        n_statistics_warnings = 0
        n_negative_weights_warnings = 0

        # Main loop over thetas
        for i_set in range(n_sets):

            # Setup for set
            n_samples = n_samples_per_theta[i_set]

            theta_types = [t[i_set] for t in theta_sets_types]
            theta_values = [t[i_set] for t in theta_sets_values]

            if self.morpher is None and "morphing" in theta_types:
                raise RuntimeError("Theta defined through morphing, but no morphing setup has been loaded.")

            # Parse thetas and calculate the w_c(theta) for them
            thetas = []
            theta_matrices = []
            theta_gradient_matrices = []

            logger.debug("Drawing %s events for the following thetas:", n_samples)

            for i_theta, (theta_type, theta_value) in enumerate(zip(theta_types, theta_values)):
                theta = get_theta_value(theta_type, theta_value, self.benchmarks)
                theta = np.broadcast_to(theta, (n_samples, theta.size))
                thetas.append(theta)

                theta_matrices.append(
                    get_theta_benchmark_matrix(theta_type, theta_value, self.benchmarks, self.morpher)
                )
                if needs_gradients:
                    theta_gradient_matrices.append(
                        get_dtheta_benchmark_matrix(theta_type, theta_value, self.benchmarks, self.morpher)
                    )

                logger.debug(
                    "  theta %s = %s%s", i_theta, theta[0, :], " (sampling)" if i_theta == sampling_theta_index else ""
                )

            sampling_theta_matrix = theta_matrices[sampling_theta_index]

            # Total xsec for sampling theta
            xsec_sampling_theta = mdot(sampling_theta_matrix, xsecs_benchmarks)
            rms_xsec_sampling_theta = (
                mdot(sampling_theta_matrix * sampling_theta_matrix, squared_weight_sum_benchmarks)
            ) ** 0.5

            if rms_xsec_sampling_theta > 0.1 * xsec_sampling_theta:
                n_statistics_warnings += 1

                if n_statistics_warnings <= 1:
                    logger.warning(
                        "Large statistical uncertainty on the total cross section for theta = %s: "
                        "(%4f +/- %4f) pb. Skipping these warnings in the future...",
                        thetas[sampling_theta_index][0],
                        xsec_sampling_theta,
                        rms_xsec_sampling_theta,
                    )

            # Prepare output
            samples_done = np.zeros(n_samples, dtype=np.bool)
            samples_x = np.zeros((n_samples, n_observables))
            samples_augmented_data = []
            for definition in augmented_data_definitions:
                if definition[0] == "ratio":
                    samples_augmented_data.append(np.zeros((n_samples, 1)))
                elif definition[0] == "score":
                    samples_augmented_data.append(np.zeros((n_samples, self.n_parameters)))
                elif definition[0] == "nuisance_score":
                    samples_augmented_data.append(np.zeros((n_samples, self.n_nuisance_parameters)))

            largest_weight = 0.0

            # Main sampling loop
            while not np.all(samples_done):

                # Draw random numbers in [0, 1]
                u = np.random.rand(n_samples)  # Shape: (n_samples,)

                # Loop over weighted events
                cumulative_p = np.array([0.0])

                for x_batch, weights_benchmarks_batch in madminer_event_loader(
                    self.madminer_filename, start=start_event, end=end_event
                ):
                    # Evaluate p(x | sampling theta)
                    weights_theta = mdot(sampling_theta_matrix, weights_benchmarks_batch)  # Shape (n_batch_size,)
                    p_theta = weights_theta / xsec_sampling_theta  # Shape: (n_batch_size,)

                    # Handle negative weights (should be rare)
                    n_negative_weights = np.sum(p_theta < 0.0)
                    if n_negative_weights > 0:
                        n_negative_weights_warnings += 1
                        # n_negative_benchmark_weights = np.sum(weights_benchmarks_batch < 0.0)

                        if n_negative_weights_warnings <= 3:
                            logger.warning(
                                "For this value of theta, %s / %s events have negative weight and will be ignored",
                                n_negative_weights,
                                p_theta.size,
                            )
                            if n_negative_weights_warnings == 3:
                                logger.warning("Skipping warnings about negative weights in the future...")

                        # filter_negative_weights = p_theta < 0.0
                        # for weight_theta_neg, weight_benchmarks_neg in zip(
                        #     weights_theta[filter_negative_weights], weights_benchmarks_batch[filter_negative_weights]
                        # ):
                        #     logger.debug(
                        #         "  weight(theta): %s, benchmark weights: %s", weight_theta_neg, weight_benchmarks_neg
                        #     )

                    p_theta[p_theta < 0.0] = 0.0

                    # Remember largest weights (to calculate effective number of samples)
                    largest_weight = max(largest_weight, np.max(p_theta))

                    # Calculate cumulative p (summing up all events until here)
                    cumulative_p = cumulative_p.flatten()[-1] + np.cumsum(p_theta)  # Shape: (n_batch_size,)

                    # When cumulative_p hits u, we store the events
                    indices = np.searchsorted(cumulative_p, u, side="left").flatten()
                    # Shape: (n_samples,), values: [0, ..., n_batch_size]

                    found_now = np.invert(samples_done) & (indices < len(cumulative_p))  # Shape: (n_samples,)
                    samples_x[found_now] = x_batch[indices[found_now]]
                    samples_done[found_now] = True

                    # Extract augmented data
                    relevant_augmented_data = calculate_augmented_data(
                        augmented_data_definitions,
                        weights_benchmarks_batch[indices[found_now], :],
                        xsecs_benchmarks,
                        theta_matrices,
                        theta_gradient_matrices,
                        nuisance_morpher=self.nuisance_morpher,
                    )

                    for i, this_relevant_augmented_data in enumerate(relevant_augmented_data):
                        samples_augmented_data[i][found_now] = this_relevant_augmented_data

                    if np.all(samples_done):
                        break

                # Cross-check cumulative probabilities at end
                logger.debug("  Cumulative probability (should be close to 1): %s", cumulative_p[-1])

                # Check that we got 'em all, otherwise repeat
                if not np.all(samples_done):
                    logger.debug(
                        "  After full pass through event files, {} / {} samples not found, u = {}".format(
                            np.sum(np.invert(samples_done)), samples_done.size, u[np.invert(samples_done)]
                        )
                    )

            all_x.append(samples_x)
            for i, theta in enumerate(thetas):
                all_thetas[i].append(theta)
            for i, this_samples_augmented_data in enumerate(samples_augmented_data):
                all_augmented_data[i].append(this_samples_augmented_data)
            all_effective_n_samples.append(1.0 / max(1.0e-12, largest_weight))

        # Combine and return results
        all_x = np.vstack(all_x)
        for i in range(n_thetas):
            all_thetas[i] = np.vstack(all_thetas[i])
        for i in range(len(all_augmented_data)):
            all_augmented_data[i] = np.vstack(all_augmented_data[i])
        all_effective_n_samples = np.array(all_effective_n_samples)

        # Report effective number of samples
        if n_sets > 1:
            logger.info(
                "Effective number of samples: mean %s, with individual thetas ranging from %s to %s",
                np.mean(all_effective_n_samples),
                np.min(all_effective_n_samples),
                np.max(all_effective_n_samples),
            )
            logger.debug("Effective number of samples for all thetas: %s", all_effective_n_samples)
        else:
            logger.info("Effective number of samples: %s", all_effective_n_samples[0])

        return all_x, all_augmented_data, all_thetas

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
