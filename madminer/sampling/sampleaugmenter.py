import time
import logging
import numpy as np
import multiprocessing

from functools import partial
from pathlib import Path

from ..analysis import DataAnalyzer
from ..utils.various import shuffle

logger = logging.getLogger(__name__)


class SampleAugmenter(DataAnalyzer):
    """
    Sampling / unweighting and data augmentation.

    After the generated events have been analyzed and the observables and weights have been saved into a MadMiner file,
    for instance with `madminer.delphes.DelphesReader` or `madminer.lhe.LHEReader`, the next step is typically
    the generation of training and evaluation data for the machine learning algorithms. This generally involves two
    (related) tasks: unweighting, i.e. the creation of samples that do not carry individual weights but follow some
    distribution, and the extraction of the joint likelihood ratio and / or joint score (the "augmented data").

    After initializing `SampleAugmenter` with the filename of a MadMiner file, this is done with a single function call.
    Depending on the downstream inference algorithm, there are different possibilities:

    * `SampleAugmenter.sample_train_plain()` creates plain training samples without augmented data.
    * `SampleAugmenter.sample_train_local()` creates training samples for local methods based on the score,
      such as SALLY and SALLINO.
    * `SampleAugmenter.sample_train_density()` creates training samples for non-local methods based on density
      estimation and the score, such as SCANDAL.
    * `SampleAugmenter.sample_train_ratio()` creates training samples for non-local, ratio-based methods
      like RASCAL or ALICE.
    * `SampleAugmenter.sample_train_more_ratios()` does the same, but can extract joint ratios and scores
      at more parameter points. This additional information  can be used efficiently in the setup with a "doubly
      parameterized" likelihood ratio estimator that models the dependence on both the numerator and denominator
      hypothesis.
    * `SampleAugmenter.sample_test()` creates evaluation samples for all methods.

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
        super().__init__(filename, disable_morphing, include_nuisance_parameters)

    def sample_train_plain(
        self,
        theta,
        n_samples,
        nu=None,
        sample_only_from_closest_benchmark=True,
        folder=None,
        filename=None,
        test_split=0.2,
        validation_split=0.2,
        partition="train",
        n_processes=1,
        n_eff_forced=None,
        double_precision=False,
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

        nu : None or tuple, optional
            Tuple (type, value) that defines the nuisance parameter point or prior over parameter points for the
            sampling. Default value: None

        sample_only_from_closest_benchmark : bool, optional
            If True, only weighted events originally generated from the closest benchmarks are used. Default value: True.

        folder : str or None
            Path to the folder where the resulting samples should be saved (ndarrays in .npy format). Default value:
            None.

        filename : str or None
            Filenames for the resulting samples. A prefix such as 'x' or 'theta0' as well as the extension
            '.npy' will be added automatically. Default value:
            None.

        test_split : float or None, optional
            Fraction of events reserved for the evaluation sample (that will not be used for any training samples).
            Default value: 0.2.

        validation_split : float or None, optional
            Fraction of events reserved for testing. Default value: 0.2.

        partition : {"train", "test", "validation", "all"}, optional
            Which event partition to use. Default value: "train".

        n_processes : None or int, optional
            If None or larger than 1, MadMiner will use multiprocessing to parallelize the sampling. In this case,
            n_workers sets the number of jobs running in parallel, and None will use the number of CPUs. Default value:
            1.

        n_eff_forced : float, optional
            If not None, MadMiner will require the relative weights of the events to be smaller than 1/n_eff_forced
            and ignore other events. This can help to reduce statistical effects caused by a small number of events
            with very large weights obtained by the morphing procedure. Default value: None

        double_precision : bool, optional
            Use double floating-point precision. Default value: False.


        Returns
        -------
        x : ndarray
            Observables with shape `(n_samples, n_observables)`. The same information is saved as a file in the given
            folder.

        theta : ndarray
            Parameter points used for sampling with shape `(n_samples, n_parameters)`. The same information is saved as
            a file in the given folder.

        effective_n_samples : int
            Effective number of samples, defined as 1/max(event_probabilities), where event_probabilities are the
            fractions of the cross section carried by each event.

        """

        logger.info("Extracting plain training sample. Sampling according to %s", self._format_sampling(theta))

        # Parameters
        parsed_thetas, n_samples_per_theta = self._parse_theta(theta, n_samples)
        parsed_nus = self._parse_nu(nu, len(parsed_thetas))
        sets = self._build_sets([parsed_thetas], [parsed_nus])

        # Start
        x, _, (theta,), effective_n_samples = self._sample(
            sets=sets,
            n_samples_per_set=n_samples_per_theta,
            partition=partition,
            validation_split=validation_split,
            test_split=test_split,
            n_processes=n_processes,
            sample_only_from_closest_benchmark=sample_only_from_closest_benchmark,
            n_eff_forced=n_eff_forced,
            double_precision=double_precision,
        )

        # Save data
        if filename is not None and folder is not None:
            Path(folder).mkdir(parents=True, exist_ok=True)
            np.save(f"{folder}/theta_{filename}.npy", theta)
            np.save(f"{folder}/x_{filename}.npy", x)

        return x, theta, min(effective_n_samples)

    def sample_train_local(
        self,
        theta,
        n_samples,
        nu=None,
        sample_only_from_closest_benchmark=True,
        folder=None,
        filename=None,
        nuisance_score="auto",
        test_split=0.2,
        validation_split=0.2,
        partition="train",
        n_processes=1,
        log_message=True,
        n_eff_forced=None,
        double_precision=False,
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

        nu : None or tuple, optional
            Tuple (type, value) that defines the nuisance parameter point or prior over parameter points for the
            sampling. Default value: None

        sample_only_from_closest_benchmark : bool, optional
            If True, only weighted events originally generated from the closest benchmarks are used. Default value: True.

        folder : str or None
            Path to the folder where the resulting samples should be saved (ndarrays in .npy format). Default value:
            None.

        filename : str or None
            Filenames for the resulting samples. A prefix such as 'x' or 'theta0' as well as the extension
            '.npy' will be added automatically. Default value:
            None.

        nuisance_score : bool or "auto", optional
            If True, the score with respect to the nuisance parameters (at the default position) will also be
            calculated. If False, only the score with respect to the physics parameters is calculated. For "auto",
            the nuisance score will be calculated if a nuisance setup is defined. Default: True.

        test_split : float or None, optional
            Fraction of events reserved for the evaluation sample (that will not be used for any training samples).
            Default value: 0.2.

        validation_split : float or None, optional
            Fraction of events reserved for testing. Default value: 0.2.

        partition : {"train", "test", "validation", "all"}, optional
            Which event partition to use. Default value: "train".

        n_processes : None or int, optional
            If None or larger than 1, MadMiner will use multiprocessing to parallelize the sampling. In this case,
            n_workers sets the number of jobs running in parallel, and None will use the number of CPUs. Default value:
            1.

        log_message : bool, optional
            If True, logging output. This option is only designed for internal use.

        n_eff_forced : float, optional
            If not None, MadMiner will require the relative weights of the events to be smaller than 1/n_eff_forced
            and ignore other events. This can help to reduce statistical effects caused by a small number of events
            with very large weights obtained by the morphing procedure. Default value: None

        double_precision : bool, optional
            Use double floating-point precision. Default value: False.

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

        effective_n_samples : int
            Effective number of samples, defined as 1/max(event_probabilities), where event_probabilities are the
            fractions of the cross section carried by each event.

        """

        if log_message:
            logger.info(
                "Extracting training sample for local score regression. "
                "Sampling and score evaluation according to %s",
                self._format_sampling(theta),
            )

        # Check setup
        if nuisance_score == "auto":
            nuisance_score = self.nuisance_morpher is not None
        if self.morpher is None and self.finite_difference_benchmarks is None:
            raise RuntimeError("Neither morphing setup nor finite-difference setup loaded. Cannot calculate score.")
        if self.nuisance_morpher is None and nuisance_score:
            raise RuntimeError("No nuisance parameters defined. Cannot calculate nuisance score.")

        # Parameters
        parsed_thetas, n_samples_per_theta = self._parse_theta(theta, n_samples)
        parsed_nus = self._parse_nu(nu, len(parsed_thetas))
        sets = self._build_sets([parsed_thetas], [parsed_nus])

        # Augmented data (gold)
        augmented_data_definitions = [("score", 0)]

        # Start
        x, augmented_data, (theta,), effective_n_samples = self._sample(
            sets=sets,
            n_samples_per_set=n_samples_per_theta,
            augmented_data_definitions=augmented_data_definitions,
            nuisance_score=nuisance_score,
            partition=partition,
            validation_split=validation_split,
            test_split=test_split,
            n_processes=n_processes,
            sample_only_from_closest_benchmark=sample_only_from_closest_benchmark,
            n_eff_forced=n_eff_forced,
            double_precision=double_precision,
        )
        t_xz = augmented_data[0]

        # Save data
        if filename is not None and folder is not None:
            Path(folder).mkdir(parents=True, exist_ok=True)
            np.save(f"{folder}/theta_{filename}.npy", theta)
            np.save(f"{folder}/x_{filename}.npy", x)
            np.save(f"{folder}/t_xz_{filename}.npy", t_xz)

        return x, theta, t_xz, min(effective_n_samples)

    def sample_train_density(
        self,
        theta,
        n_samples,
        nu=None,
        sample_only_from_closest_benchmark=True,
        folder=None,
        filename=None,
        nuisance_score="auto",
        test_split=0.2,
        validation_split=0.2,
        partition="train",
        n_processes=1,
        n_eff_forced=None,
        double_precision=False,
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

        nu : None or tuple, optional
            Tuple (type, value) that defines the nuisance parameter point or prior over parameter points for the
            sampling. Default value: None

        sample_only_from_closest_benchmark : bool, optional
            If True, only weighted events originally generated from the closest benchmarks are used. Default value: True.

        folder : str or None
            Path to the folder where the resulting samples should be saved (ndarrays in .npy format). Default value:
            None.

        filename : str or None
            Filenames for the resulting samples. A prefix such as 'x' or 'theta0' as well as the extension
            '.npy' will be added automatically. Default value: None.

        nuisance_score : bool or "auto", optional
            If True, the score with respect to the nuisance parameters (at the default position) will also be
            calculated. If False, only the score with respect to the physics parameters is calculated. For "auto",
            the nuisance score will be calculated if a nuisance setup is defined. Default: True.

        test_split : float or None, optional
            Fraction of events reserved for the evaluation sample (that will not be used for any training samples).
            Default value: 0.2.

        validation_split : float or None, optional
            Fraction of events reserved for testing. Default value: 0.2.

        partition : {"train", "test", "validation", "all"}, optional
            Which event partition to use. Default value: "train".

        n_processes : None or int, optional
            If None or larger than 1, MadMiner will use multiprocessing to parallelize the sampling. In this case,
            n_workers sets the number of jobs running in parallel, and None will use the number of CPUs. Default value:
            1.

        n_eff_forced : float, optional
            If not None, MadMiner will require the relative weights of the events to be smaller than 1/n_eff_forced
            and ignore other events. This can help to reduce statistical effects caused by a small number of events
            with very large weights obtained by the morphing procedure. Default value: None

        double_precision : bool, optional
            Use double floating-point precision. Default value: False.

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

        effective_n_samples : int
            Effective number of samples, defined as 1/max(event_probabilities), where event_probabilities are the
            fractions of the cross section carried by each event.

        """

        logger.info(
            "Extracting training sample for non-local score-based methods. "
            "Sampling and score evaluation according to %s",
            theta,
        )

        return self.sample_train_local(
            theta=theta,
            n_samples=n_samples,
            nu=nu,
            sample_only_from_closest_benchmark=sample_only_from_closest_benchmark,
            folder=folder,
            filename=filename,
            nuisance_score=nuisance_score,
            partition=partition,
            validation_split=validation_split,
            test_split=test_split,
            n_processes=n_processes,
            log_message=False,
            n_eff_forced=n_eff_forced,
            double_precision=double_precision,
        )

    def sample_train_ratio(
        self,
        theta0,
        theta1,
        n_samples,
        nu0=None,
        nu1=None,
        sample_only_from_closest_benchmark=True,
        folder=None,
        filename=None,
        nuisance_score="auto",
        test_split=0.2,
        validation_split=0.2,
        partition="train",
        n_processes=1,
        return_individual_n_effective=False,
        n_eff_forced=None,
        double_precision=False,
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

        nu0 : None or tuple, optional
            Tuple (type, value) that defines the numerator nuisance parameter point or prior over parameter points for
            the sampling. Default value: None

        nu1 : None or tuple, optional
            Tuple (type, value) that defines the denominator nuisance parameter point or prior over parameter points for
            the sampling. Default value: None

        sample_only_from_closest_benchmark : bool, optional
            If True, only weighted events originally generated from the closest benchmarks are used. Default value: True.

        folder : str or None
            Path to the folder where the resulting samples should be saved (ndarrays in .npy format). Default value:
            None.

        filename : str or None
            Filenames for the resulting samples. A prefix such as 'x' or 'theta0' as well as the extension
            '.npy' will be added automatically. Default value:
            None.

        nuisance_score : bool or "auto", optional
            If True, the score with respect to the nuisance parameters (at the default position) will also be
            calculated. If False, only the score with respect to the physics parameters is calculated. For "auto",
            the nuisance score will be calculated if a nuisance setup is defined. Default: True.

        test_split : float or None, optional
            Fraction of events reserved for the evaluation sample (that will not be used for any training samples).
            Default value: 0.2.

        validation_split : float or None, optional
            Fraction of events reserved for testing. Default value: 0.2.

        partition : {"train", "test", "validation", "all"}, optional
            Which event partition to use. Default value: "train".

        n_processes : None or int, optional
            If None or larger than 1, MadMiner will use multiprocessing to parallelize the sampling. In this case,
            n_workers sets the number of jobs running in parallel, and None will use the number of CPUs. Default value:
            1.

        return_individual_n_effective : bool, optional
            Returns number of effective samples for each set individually. Default value: False.

        n_eff_forced : float, optional
            If not None, MadMiner will require the relative weights of the events to be smaller than 1/n_eff_forced
            and ignore other events. This can help to reduce statistical effects caused by a small number of events
            with very large weights obtained by the morphing procedure. Default value: None

        double_precision : bool, optional
            Use double floating-point precision. Default value: False

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

        effective_n_samples : int
            Effective number of samples, defined as 1/max(event_probabilities), where event_probabilities are the
            fractions of the cross section carried by each event.

        """

        logger.info(
            "Extracting training sample for ratio-based methods. "
            "Numerator hypothesis: %s, denominator hypothesis: %s",
            self._format_sampling(theta0),
            self._format_sampling(theta1),
        )

        # Check setup
        if nuisance_score == "auto":
            nuisance_score = self.nuisance_morpher is not None
        if self.morpher is None and self.finite_difference_benchmarks is None:
            raise RuntimeError("Neither morphing setup nor finite-difference setup loaded. Cannot calculate score.")
        if self.nuisance_morpher is None and nuisance_score:
            raise RuntimeError("No nuisance parameters defined. Cannot calculate nuisance score.")

        # Augmented data (gold)
        augmented_data_definitions = [("ratio", 0, 1)]
        if self.morpher is not None:
            augmented_data_definitions.append(("score", 0))

        # Thetas for theta0 sampling
        parsed_theta0s, n_samples_per_theta0 = self._parse_theta(theta0, n_samples // 2)
        parsed_theta1s, n_samples_per_theta1 = self._parse_theta(theta1, n_samples // 2)
        parsed_nu0s = self._parse_nu(nu0, len(parsed_theta0s))
        parsed_nu1s = self._parse_nu(nu1, len(parsed_theta1s))
        sets = self._build_sets([parsed_theta0s, parsed_theta1s], [parsed_nu0s, parsed_nu1s])

        n_samples_per_theta = min(n_samples_per_theta0, n_samples_per_theta1)

        # Start for theta0
        if self.morpher is None:
            x0, (r_xz0,), (theta0_0, theta1_0), n_effective_samples_0 = self._sample(
                sets=sets,
                sampling_index=0,
                n_samples_per_set=n_samples_per_theta,
                augmented_data_definitions=augmented_data_definitions,
                nuisance_score=nuisance_score,
                partition=partition,
                validation_split=validation_split,
                test_split=test_split,
                n_processes=n_processes,
                sample_only_from_closest_benchmark=sample_only_from_closest_benchmark,
                n_eff_forced=n_eff_forced,
                double_precision=double_precision,
            )
            t_xz0 = None
        else:
            x0, (r_xz0, t_xz0), (theta0_0, theta1_0), n_effective_samples_0 = self._sample(
                sets=sets,
                sampling_index=0,
                n_samples_per_set=n_samples_per_theta,
                augmented_data_definitions=augmented_data_definitions,
                nuisance_score=nuisance_score,
                partition=partition,
                validation_split=validation_split,
                test_split=test_split,
                n_processes=n_processes,
                sample_only_from_closest_benchmark=sample_only_from_closest_benchmark,
                n_eff_forced=n_eff_forced,
                double_precision=double_precision,
            )

        # Thetas for theta1 sampling (could be different if num or denom are random)
        parsed_theta0s, n_samples_per_theta0 = self._parse_theta(theta0, n_samples // 2)
        parsed_theta1s, n_samples_per_theta1 = self._parse_theta(theta1, n_samples // 2)
        parsed_nu0s = self._parse_nu(nu0, len(parsed_theta0s))
        parsed_nu1s = self._parse_nu(nu1, len(parsed_theta1s))
        sets = self._build_sets([parsed_theta0s, parsed_theta1s], [parsed_nu0s, parsed_nu1s])

        n_samples_per_theta = min(n_samples_per_theta0, n_samples_per_theta1)

        # Start for theta1
        if self.morpher is None:
            x1, (r_xz1,), (theta0_1, theta1_1), n_effective_samples_1 = self._sample(
                sets=sets,
                sampling_index=1,
                n_samples_per_set=n_samples_per_theta,
                augmented_data_definitions=augmented_data_definitions,
                nuisance_score=nuisance_score,
                partition=partition,
                validation_split=validation_split,
                test_split=test_split,
                n_processes=n_processes,
                sample_only_from_closest_benchmark=sample_only_from_closest_benchmark,
                n_eff_forced=n_eff_forced,
                double_precision=double_precision,
            )
            t_xz1 = None
        else:
            x1, (r_xz1, t_xz1), (theta0_1, theta1_1), n_effective_samples_1 = self._sample(
                sets=sets,
                sampling_index=1,
                n_samples_per_set=n_samples_per_theta,
                augmented_data_definitions=augmented_data_definitions,
                nuisance_score=nuisance_score,
                partition=partition,
                validation_split=validation_split,
                test_split=test_split,
                n_processes=n_processes,
                sample_only_from_closest_benchmark=sample_only_from_closest_benchmark,
                n_eff_forced=n_eff_forced,
                double_precision=double_precision,
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
        n_effective = np.hstack((n_effective_samples_0, n_effective_samples_1))

        # Shuffle
        x, r_xz, t_xz, theta0, theta1, y, n_effective = shuffle(x, r_xz, t_xz, theta0, theta1, y, n_effective)

        # y shape
        y = y.reshape((-1, 1))

        # Save data
        if filename is not None and folder is not None:
            Path(folder).mkdir(parents=True, exist_ok=True)
            np.save(f"{folder}/theta0_{filename}.npy", theta0)
            np.save(f"{folder}/theta1_{filename}.npy", theta1)
            np.save(f"{folder}/x_{filename}.npy", x)
            np.save(f"{folder}/y_{filename}.npy", y)
            np.save(f"{folder}/r_xz_{filename}.npy", r_xz)
            if self.morpher is not None:
                np.save(f"{folder}/t_xz_{filename}.npy", t_xz)

        if not return_individual_n_effective:
            n_effective = np.min(n_effective)

        return x, theta0, theta1, y, r_xz, t_xz, n_effective

    def sample_train_more_ratios(
        self,
        theta0,
        theta1,
        n_samples,
        nu0=None,
        nu1=None,
        sample_only_from_closest_benchmark=True,
        folder=None,
        filename=None,
        additional_thetas=None,
        nuisance_score="auto",
        test_split=0.2,
        validation_split=0.2,
        partition="train",
        n_processes=1,
        n_eff_forced=None,
        double_precision=False,
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

        nu0 : None or tuple, optional
            Tuple (type, value) that defines the numerator nuisance parameter point or prior over parameter points for
            the sampling. Default value: None

        nu1 : None or tuple, optional
            Tuple (type, value) that defines the denominator nuisance parameter point or prior over parameter points for
            the sampling. Default value: None

        sample_only_from_closest_benchmark : bool, optional
            If True, only weighted events originally generated from the closest benchmarks are used. Default value: True.

        folder : str or None
            Path to the folder where the resulting samples should be saved (ndarrays in .npy format). Default value:
            None.

        filename : str or None
            Filenames for the resulting samples. A prefix such as 'x' or 'theta0' as well as the extension
            '.npy' will be added automatically. Default value:
            None.

        additional_thetas : list of tuple or None
            list of tuples `(type, value)` that defines additional theta points at which ratio and score are evaluated,
            and which are then used to create additional training data points. These can be efficiently used only in
            the "doubly parameterized" setup where a likelihood ratio estimator models the dependence of the likelihood
            ratio on both the numerator and denominator hypothesis. Pass the output of  the helper functions
            `constant_benchmark_theta()`, `multiple_benchmark_thetas()`, `constant_morphing_theta()`,
            `multiple_morphing_thetas()`, or `random_morphing_thetas()`. Default value: None.

        nuisance_score : bool or "auto", optional
            If True, the score with respect to the nuisance parameters (at the default position) will also be
            calculated. If False, only the score with respect to the physics parameters is calculated. For "auto",
            the nuisance score will be calculated if a nuisance setup is defined. Default: True.

        test_split : float or None, optional
            Fraction of events reserved for the evaluation sample (that will not be used for any training samples).
            Default value: 0.2.

        validation_split : float or None, optional
            Fraction of events reserved for testing. Default value: 0.2.

        partition : {"train", "test", "validation", "all"}, optional
            Which event partition to use. Default value: "train".

        n_processes : None or int, optional
            If None or larger than 1, MadMiner will use multiprocessing to parallelize the sampling. In this case,
            n_workers sets the number of jobs running in parallel, and None will use the number of CPUs. Default value:
            1.

        n_eff_forced : float, optional
            If not None, MadMiner will require the relative weights of the events to be smaller than 1/n_eff_forced
            and ignore other events. This can help to reduce statistical effects caused by a small number of events
            with very large weights obtained by the morphing procedure. Default value: None

        double_precision : bool, optional
            Use double floating-point precision. Default value: False

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

        effective_n_samples : int
            Effective number of samples, defined as 1/max(event_probabilities), where event_probabilities are the
            fractions of the cross section carried by each event.

        """

        logger.info(
            "Extracting training sample for ratio-based methods. "
            "Numerator hypothesis: %s, denominator hypothesis: %s",
            self._format_sampling(theta0),
            self._format_sampling(theta1),
        )

        # Check setup
        if nuisance_score == "auto":
            nuisance_score = self.nuisance_morpher is not None
        if self.morpher is None and self.finite_difference_benchmarks is None:
            raise RuntimeError("Neither morphing setup nor finite-difference setup loaded. Cannot calculate score.")
        if self.nuisance_morpher is None and nuisance_score:
            raise RuntimeError("No nuisance parameters defined. Cannot calculate nuisance score.")
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

        # Parse thetas for theta0 sampling
        parsed_thetas = []
        parsed_nus = []
        n_samples_per_theta = 1000000

        parsed_theta0s, this_n_samples = self._parse_theta(theta0, n_samples // 2)
        parsed_nu0s = self._parse_nu(nu0, len(parsed_theta0s))
        parsed_thetas.append(parsed_theta0s)
        parsed_nus.append(parsed_nu0s)
        n_samples_per_theta = min(this_n_samples, n_samples_per_theta)

        parsed_theta1s, this_n_samples = self._parse_theta(theta1, n_samples // 2)
        parsed_nu1s = self._parse_nu(nu1, len(parsed_theta1s))
        parsed_thetas.append(parsed_theta1s)
        parsed_nus.append(parsed_nu1s)
        n_samples_per_theta = min(this_n_samples, n_samples_per_theta)

        for additional_theta in additional_thetas:
            additional_parsed_thetas, this_n_samples = self._parse_theta(additional_theta, n_samples // 2)
            parsed_thetas.append(additional_parsed_thetas)
            additional_parsed_nu = self._parse_nu(nu1, len(additional_parsed_thetas))
            parsed_nus.append(additional_parsed_nu)
            n_samples_per_theta = min(this_n_samples, n_samples_per_theta)

        sets = self._build_sets(parsed_thetas, parsed_nus)

        # Start for theta0
        x_0, augmented_data_0, thetas_0, n_effective_samples_0 = self._sample(
            sets=sets,
            n_samples_per_set=n_samples_per_theta,
            augmented_data_definitions=augmented_data_definitions_0,
            sampling_index=0,
            nuisance_score=nuisance_score,
            partition=partition,
            validation_split=validation_split,
            test_split=test_split,
            n_processes=n_processes,
            sample_only_from_closest_benchmark=sample_only_from_closest_benchmark,
            n_eff_forced=n_eff_forced,
            double_precision=double_precision,
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
        parsed_thetas = []
        parsed_nus = []
        n_samples_per_theta = 1000000

        parsed_thetas0, this_n_samples = self._parse_theta(theta0, n_samples // 2)
        parsed_nu0s = self._parse_nu(nu0, len(parsed_theta0s))
        parsed_thetas.append(parsed_thetas0)
        parsed_nus.append(parsed_nu0s)
        n_samples_per_theta = min(this_n_samples, n_samples_per_theta)

        parsed_thetas1, this_n_samples = self._parse_theta(theta1, n_samples // 2)
        parsed_nu1s = self._parse_nu(nu1, len(parsed_theta1s))
        parsed_thetas.append(parsed_thetas1)
        parsed_nus.append(parsed_nu1s)
        n_samples_per_theta = min(this_n_samples, n_samples_per_theta)

        for additional_theta in additional_thetas:
            additional_parsed_thetas, this_n_samples = self._parse_theta(additional_theta, n_samples // 2)
            additional_parsed_nu = self._parse_nu(nu0, len(additional_parsed_thetas))
            parsed_thetas.append(additional_parsed_thetas)
            parsed_nus.append(additional_parsed_nu)
            n_samples_per_theta = min(this_n_samples, n_samples_per_theta)

        sets = self._build_sets(parsed_thetas, parsed_nus)

        # Start for theta1
        x_1, augmented_data_1, thetas_1, n_effective_samples_1 = self._sample(
            sets=sets,
            n_samples_per_set=n_samples_per_theta,
            augmented_data_definitions=augmented_data_definitions_1,
            sampling_index=1,
            nuisance_score=nuisance_score,
            partition=partition,
            validation_split=validation_split,
            test_split=test_split,
            n_processes=n_processes,
            sample_only_from_closest_benchmark=sample_only_from_closest_benchmark,
            n_eff_forced=n_eff_forced,
            double_precision=double_precision,
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
            Path(folder).mkdir(parents=True, exist_ok=True)
            np.save(f"{folder}/theta0_{filename}.npy", theta0)
            np.save(f"{folder}/theta1_{filename}.npy", theta1)
            np.save(f"{folder}/x_{filename}.npy", x)
            np.save(f"{folder}/y_{filename}.npy", y)
            np.save(f"{folder}/r_xz_{filename}.npy", r_xz)
            np.save(f"{folder}/t_xz0_{filename}.npy", t_xz0)
            np.save(f"{folder}/t_xz1_{filename}.npy", t_xz1)

        return x, theta0, theta1, y, r_xz, t_xz0, t_xz1, min(min(n_effective_samples_0), min(n_effective_samples_1))

    def sample_test(
        self,
        theta,
        n_samples,
        nu=None,
        sample_only_from_closest_benchmark=True,
        folder=None,
        filename=None,
        test_split=0.2,
        validation_split=0.2,
        partition="test",
        n_processes=1,
        n_eff_forced=None,
        double_precision=False,
    ):
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

        nu : None or tuple, optional
            Tuple (type, value) that defines the nuisance parameter point or prior over parameter points for the
            sampling. Default value: None

        sample_only_from_closest_benchmark : bool, optional
            If True, only weighted events originally generated from the closest benchmarks are used. Default value: True.

        folder : str or None
            Path to the folder where the resulting samples should be saved (ndarrays in .npy format). Default value:
            None.

        filename : str or None
            Filenames for the resulting samples. A prefix such as 'x' or 'theta0' as well as the extension
            '.npy' will be added automatically. Default value:
            None.

        test_split : float or None, optional
            Fraction of events reserved for the evaluation sample (that will not be used for any training samples).
            Default value: 0.2.

        validation_split : float or None, optional
            Fraction of events reserved for testing. Default value: 0.2.

        partition : {"train", "test", "validation", "all"}, optional
            Which event partition to use. Default value: "test".

        n_processes : None or int, optional
            If None or larger than 1, MadMiner will use multiprocessing to parallelize the sampling. In this case,
            n_workers sets the number of jobs running in parallel, and None will use the number of CPUs. Default value:
            1.

        n_eff_forced : float, optional
            If not None, MadMiner will require the relative weights of the events to be smaller than 1/n_eff_forced
            and ignore other events. This can help to reduce statistical effects caused by a small number of events
            with very large weights obtained by the morphing procedure. Default value: None

        double_precision : bool, optional
            Use double floating-point precision. Default value: False

        Returns
        -------
        x : ndarray
            Observables with shape `(n_samples, n_observables)`. The same information is saved as a file in the given
            folder.

        theta : ndarray
            Parameter points used for sampling with shape `(n_samples, n_parameters)`. The same information is saved as
            a file in the given folder.

        effective_n_samples : int
            Effective number of samples, defined as 1/max(event_probabilities), where event_probabilities are the
            fractions of the cross section carried by each event.

        """

        logger.info("Extracting evaluation sample. Sampling according to %s", self._format_sampling(theta))

        # Thetas
        parsed_thetas, n_samples_per_theta = self._parse_theta(theta, n_samples)
        parsed_nus = self._parse_nu(nu, len(parsed_thetas))
        sets = self._build_sets([parsed_thetas], [parsed_nus])

        # Extract information
        x, _, (theta,), n_effective_samples = self._sample(
            sets=sets,
            n_samples_per_set=n_samples_per_theta,
            partition=partition,
            validation_split=validation_split,
            test_split=test_split,
            n_processes=n_processes,
            sample_only_from_closest_benchmark=sample_only_from_closest_benchmark,
            n_eff_forced=n_eff_forced,
            double_precision=double_precision,
        )

        # Save data
        if filename is not None and folder is not None:
            Path(folder).mkdir(parents=True, exist_ok=True)
            np.save(f"{folder}/theta_{filename}.npy", theta)
            np.save(f"{folder}/x_{filename}.npy", x)

        return x, theta, min(n_effective_samples)

    def cross_sections(self, theta, nu=None):

        """
        Calculates the total cross sections for all specified thetas.

        Parameters
        ----------
        theta : tuple
            Tuple (type, value) that defines the parameter point or prior over parameter points at which the cross
            section is calculated. Pass the output of the functions `benchmark()`,
            `benchmarks()`, `morphing_point()`, `morphing_points()`, or
            `random_morphing_points()`.

        nu : tuple or None, optional
            Tuple (type, value) that defines the nuisance parameter point or prior over nuisance parameter points at
            which the cross section is calculated. Pass the output of the functions `benchmark()`,
            `benchmarks()`, `morphing_point()`, `morphing_points()`, or
            `random_morphing_points()`. Default value: None.

        Returns
        -------
        thetas : ndarray
            Parameter points with shape `(n_thetas, n_parameters)` or
            `(n_thetas, n_parameters + n_nuisance_parameters)`.

        xsecs : ndarray
            Total cross sections in pb with shape `(n_thetas, )`.

        xsec_uncertainties : ndarray
            Statistical uncertainties on the total cross sections in pb with shape `(n_thetas, )`.

        """
        logger.info("Starting cross-section calculation")
        parsed_thetas, _ = self._parse_theta(theta, None)
        theta_values = np.asarray([self._get_theta_value(parsed_theta) for parsed_theta in parsed_thetas])

        if nu is not None:
            parsed_nus = self._parse_nu(nu, len(parsed_thetas))
            nu_values = np.asarray([self._get_nu_value(parsed_nu for parsed_nu in parsed_nus)])
            param_values = np.hstack((theta_values, nu_values))
        else:
            parsed_nus = None
            param_values = theta_values

        xsecs, uncertainties = self.xsecs(thetas=parsed_thetas, nus=parsed_nus)

        return param_values, xsecs, uncertainties

    def _sample(
        self,
        sets,
        n_samples_per_set,
        sampling_index=0,
        sample_only_from_closest_benchmark=True,
        augmented_data_definitions=None,
        nuisance_score=True,
        partition="train",
        test_split=0.2,
        validation_split=0.2,
        verbose="some",
        n_processes=1,
        update_patience=0.01,
        force_update_patience=15 * 60.0,
        n_eff_forced=None,
        double_precision=False,
    ):
        """
        Low-level function for the extraction of information from the event samples. Do not use this function directly.

        The sampling is organized in terms of "sets". For each set, a number of parameter points (thetas and nus) is
        fixed, and `n_samples_per_theta` events are sampled from one of them.

        Parameters
        ----------
        sets : list of list of tuples
            The outer list goes over sets, the inner list goes over parameter points, the tuples have the form
            (theta, nu). Here theta can be a str or int (for benchmarks) or ndarray (with morphing), while nu can be
            None (for nominal value) or ndarray (for nuisance morphing).

        n_samples_per_set : int
            Number of samples to be drawn per entry in theta_sampling_types.

        sampling_index : int
            Marking the index of the theta set defined through thetas_types and
            thetas_values that should be used for sampling. Default value: 0.

        augmented_data_definitions : list of tuple or None
            Each tuple can either be ('ratio', num_theta, den_theta) or
            ('score', theta), where num_theta, den_theta, and theta are indexes marking
            which of the theta sets defined through thetas_types and thetas_values is
            used. Default value: None.

        nuisance_score : bool, optional
            If True, any joint score in the augmented data definitions is also calculated with respect to the nuisance
            parameters. Default value: True.

        partition : {"train", "test", "validation", "all"}, optional
            Which event partition to use. Default value: "train".

        test_split : float or None, optional
            Fraction of events reserved for the evaluation sample (that will not be used for any training samples).
            Default value: 0.2.

        validation_split : float or None, optional
            Fraction of events reserved for testing. Default value: 0.2.

        n_processes : None or int, optional
            If None or larger than 1, MadMiner will use multiprocessing to parallelize the sampling. In this case,
            n_workers sets the number of jobs running in parallel, and None will use the number of CPUs.
            Default value: 1.

        update_patience : float, optional
            Wait time (in s) between log update checks if n_workers > 1 (or None). Default value: 0.01

        force_update_patience : float, optional
            Wait time (in s) between log updates (independent of actual progress) if n_workers > 1 (or None). Default
            value: 15 * 60. (15 minutes).

        n_eff_forced : float, optional
            If not None, MadMiner will require the relative weights of the events to be smaller than 1/n_eff_forced
            and ignore other events. This can help to reduce statistical effects caused by a small number of events
            with very large weights obtained by the morphing procedure. Default value: None

        double_precision : bool, optional
            Use double floating-point precision. Default value: False.

        Returns
        -------
        x :  ndarray
            Observables.

        augmented_data : list of ndarray
            Augmented data.

        theta_values : list of ndarray
            Parameter values.

        """

        logger.debug("Starting sample extraction")

        if n_eff_forced is not None:
            logger.warning(
                "Trimmed sampling is turned on (n_eff_forced is not None). This option is potentially "
                "since requiring large values of n_eff_forced can significantly distort distributions."
                "Check if manually that the sampled distributions are still correct."
            )

        # Check inputs
        if augmented_data_definitions is None:
            augmented_data_definitions = []

        n_sets, n_params = self._check_sets(sets)

        # What needs to be calculated?
        needs_gradients = self._check_gradient_need(augmented_data_definitions)

        # Prepare outputs
        all_x = []
        all_augmented_data = [[] for _ in augmented_data_definitions]
        all_thetas = [[] for _ in range(n_params)]
        all_nus = [[] for _ in range(n_params)]
        all_effective_n_samples = []

        n_stats_warnings = 0
        n_neg_weights_warnings = 0
        n_too_large_weights_warnings = 0

        # Multiprocessing approach
        if n_processes is None or n_processes > 1:
            if n_processes is None:
                n_processes = multiprocessing.cpu_count()

            job = partial(
                self._sample_set,
                n_samples=n_samples_per_set,
                augmented_data_definitions=augmented_data_definitions,
                sampling_index=sampling_index,
                needs_gradients=needs_gradients,
                partition=partition,
                test_split=test_split,
                validation_split=validation_split,
                nuisance_score=nuisance_score,
                n_stats_warnings=1000,
                n_neg_weights_warnings=1000,
                sample_only_from_closest_benchmark=sample_only_from_closest_benchmark,
                n_eff_forced=n_eff_forced,
                double_precision=double_precision,
            )

            logger.info("Starting sampling jobs in parallel, using %s processes", n_processes)

            pool = multiprocessing.Pool(processes=n_processes)
            r = pool.map_async(job, sets, chunksize=1)

            next_verbose = 0
            verbose_steps = n_sets // 10
            last_update = time.time()

            while not r.ready():
                n_done = max(n_sets - r._number_left * r._chunksize, 0)
                if n_done >= next_verbose or time.time() - last_update > force_update_patience:
                    logger.info("%s / %s jobs done", max(n_sets - r._number_left * r._chunksize, 0), n_sets)
                    last_update = time.time()
                    while next_verbose <= n_done:
                        next_verbose += verbose_steps
                        time.sleep(update_patience)

            r.wait()

            logger.info("All jobs done!")

            for x, thetas, nus, augmented_data, eff_n_samples, _, _, _ in r.get():
                all_x.append(x)
                for i, values in enumerate(augmented_data):
                    all_augmented_data[i].append(values)
                for i, values in enumerate(thetas):
                    all_thetas[i].append(values)
                for i, values in enumerate(nus):
                    all_nus[i].append(values)
                all_effective_n_samples.append(eff_n_samples)

        # Serial approach
        else:
            logger.info("Starting sampling serially")

            # Verbosity
            if verbose == "all":  # Print output after every epoch
                n_sets_verbose = 1
            elif verbose == "many":  # Print output after 2%, 4%, ..., 100% progress
                n_sets_verbose = max(int(round(n_sets / 50, 0)), 1)
            elif verbose == "some":  # Print output after 10%, 20%, ..., 100% progress
                n_sets_verbose = max(int(round(n_sets / 20, 0)), 1)
            elif verbose == "few":  # Print output after 20%, 40%, ..., 100% progress
                n_sets_verbose = max(int(round(n_sets / 5, 0)), 1)
            elif verbose == "none":  # Never print output
                n_sets_verbose = n_sets + 2
            else:
                raise ValueError("Unknown value %s for keyword verbose", verbose)
            logger.debug("Will print training progress every %s sets", n_sets_verbose)

            # Loop over sets
            for i_set, set_ in enumerate(sets, start=1):
                if i_set % n_sets_verbose == 0:
                    logger.info("Sampling from parameter point %s / %s", i_set, n_sets)
                else:
                    logger.debug("Sampling from parameter point %s / %s", i_set, n_sets)

                (
                    x,
                    thetas,
                    nus,
                    augmented_data,
                    eff_n_samples,
                    n_stats_warnings,
                    n_neg_weights_warnings,
                    n_too_large_weights_warnings,
                ) = self._sample_set(
                    set_,
                    n_samples=n_samples_per_set,
                    augmented_data_definitions=augmented_data_definitions,
                    sampling_index=sampling_index,
                    needs_gradients=needs_gradients,
                    partition=partition,
                    test_split=test_split,
                    validation_split=validation_split,
                    nuisance_score=nuisance_score,
                    n_stats_warnings=n_stats_warnings,
                    n_too_large_weights_warnings=n_too_large_weights_warnings,
                    n_neg_weights_warnings=n_neg_weights_warnings,
                    sample_only_from_closest_benchmark=sample_only_from_closest_benchmark,
                    n_eff_forced=n_eff_forced,
                    double_precision=double_precision,
                )

                all_x.append(x)
                for i, values in enumerate(augmented_data):
                    all_augmented_data[i].append(values)
                for i, values in enumerate(thetas):
                    all_thetas[i].append(values)
                for i, values in enumerate(nus):
                    all_nus[i].append(values)
                all_effective_n_samples.append(eff_n_samples)

        # Combine and return results
        all_x = np.vstack(all_x)
        for i, values in enumerate(all_thetas):
            all_thetas[i] = np.vstack(values)
        for i, values in enumerate(all_nus):
            all_nus[i] = np.vstack(values)
        for i, values in enumerate(all_augmented_data):
            all_augmented_data[i] = np.vstack(values)
        all_effective_n_samples = np.hstack(all_effective_n_samples)
        all_thetas = self._combine_thetas_nus(all_thetas, all_nus)

        # Report effective number of samples
        self._report_effective_n_samples(all_effective_n_samples)

        return all_x, all_augmented_data, all_thetas, all_effective_n_samples

    @staticmethod
    def _check_sets(sets):
        n_sets = len(sets)
        n_params = None
        for set_ in sets:
            if n_params is None:
                n_params = len(set_)
            assert len(set_) == n_params
            for param_point in set_:
                assert len(param_point) == 2

        return n_sets, n_params

    @staticmethod
    def _check_gradient_need(augmented_data_definitions):
        for definition in augmented_data_definitions:
            if definition[0] == "score":
                return True
        return False

    def _sample_set(
        self,
        set_,
        n_samples,
        sample_only_from_closest_benchmark,
        augmented_data_definitions,
        sampling_index=0,
        needs_gradients=True,
        nuisance_score=True,
        partition="train",
        test_split=0.2,
        validation_split=0.2,
        n_stats_warnings=0,
        n_neg_weights_warnings=0,
        n_too_large_weights_warnings=0,
        n_eff_forced=None,
        double_precision=False,
    ):
        # Dtype
        dtype = np.float64 if double_precision else np.float32

        # Parse thetas and nus
        thetas, nus = [], []
        theta_values, nu_values = [], []
        theta_matrices, theta_gradient_matrices = [], []

        logger.debug("Drawing %s events for the following parameter points:", n_samples)

        for i_param, (theta, nu) in enumerate(set_):
            thetas.append(theta)
            nus.append(nu)

            theta_value = self._get_theta_value(theta)
            theta_value = np.broadcast_to(theta_value, (n_samples, theta_value.size)).astype(dtype)
            theta_values.append(theta_value)

            if nu is None:
                nu_value = None
                nu_values.append([[None] for _ in range(n_samples)])
            else:
                nu_value = self._get_nu_value(nu)
                nu_values.append(np.broadcast_to(nu_value, (n_samples, nu_value.size)).astype(dtype))

            theta_matrices.append(self._get_theta_benchmark_matrix(theta))
            if needs_gradients:
                theta_gradient_matrices.append(self._get_dtheta_benchmark_matrix(theta))

            if i_param == sampling_index:
                logger.debug("  %s: theta = %s, nu = %s (sampling)", i_param, theta_value[0, :], nu_value)
            else:
                logger.debug("  %s: theta = %s, nu = %s", i_param, theta_value[0, :], nu_value)

        theta_value_sampling = theta_values[sampling_index][0, :]

        # Cross sections
        xsecs, xsec_uncertainties = self.xsecs(
            thetas,
            nus,
            partition=partition,
            test_split=test_split,
            validation_split=validation_split,
            generated_close_to=None if not sample_only_from_closest_benchmark else theta_value_sampling,
        )
        if needs_gradients:
            xsec_gradients = self.xsec_gradients(
                thetas,
                nus,
                gradients="all" if nuisance_score else "theta",
                partition=partition,
                test_split=test_split,
                validation_split=validation_split,
                generated_close_to=None if not sample_only_from_closest_benchmark else theta_value_sampling,
            )
        else:
            xsec_gradients = None

        # Report large uncertainties
        if xsec_uncertainties[sampling_index] > 0.1 * xsecs[sampling_index]:
            n_stats_warnings += 1
            if n_stats_warnings <= 1:
                logger.warning(
                    "Large statistical uncertainty on the total cross section when sampling from theta = %s: "
                    "(%4f +/- %4f) pb (%s %%). Skipping these warnings in the future...",
                    theta_values[sampling_index][0],
                    xsecs[sampling_index],
                    xsec_uncertainties[sampling_index],
                    100.0 * xsec_uncertainties[sampling_index] / xsecs[sampling_index],
                )

        # Prepare output
        done = np.zeros(n_samples, dtype=bool)
        x = np.zeros((n_samples, self.n_observables), dtype=dtype)
        augmented_data = []
        for definition in augmented_data_definitions:
            if definition[0] == "ratio":
                augmented_data.append(np.zeros((n_samples, 1), dtype=dtype))
            elif definition[0] == "score":
                if nuisance_score:
                    augmented_data.append(
                        np.zeros((n_samples, self.n_parameters + self.n_nuisance_parameters), dtype=dtype)
                    )
                else:
                    augmented_data.append(np.zeros((n_samples, self.n_parameters), dtype=dtype))
        largest_event_probability = 0.0

        # Main sampling loop
        start_event, end_event, correction_factor = self._train_validation_test_split(
            partition, test_split, validation_split
        )
        logger.debug(
            "Sampling from partition %s, using weighted events %s to %s and a correction factor %s",
            partition,
            start_event,
            end_event,
            correction_factor,
        )
        while not np.all(done):
            # Draw random numbers in [0, 1]
            u = np.random.rand(n_samples)  # Shape: (n_samples,)
            cumulative_p = np.array([0.0])

            # Loop over weighted events
            for x_batch, weights_benchmarks_batch in self.event_loader(
                start=start_event,
                end=end_event,
                generated_close_to=None if not sample_only_from_closest_benchmark else theta_value_sampling,
            ):
                weights_benchmarks_batch *= correction_factor

                # Weights
                weights = self._weights(thetas, nus, weights_benchmarks_batch, theta_matrices)
                if needs_gradients:
                    weight_gradients = self._weight_gradients(
                        thetas,
                        nus,
                        weights_benchmarks_batch,
                        gradients="all" if nuisance_score else "theta",
                        theta_matrices=theta_matrices,
                        theta_gradient_matrices=theta_gradient_matrices,
                    )
                else:
                    weight_gradients = None

                # Evaluate p(x | sampling theta)
                p_sampling = weights[sampling_index] / xsecs[sampling_index]  # Shape: (n_batch_size,)

                # Handle negative weights (should be rare)
                n_negative_weights = np.sum(p_sampling < 0.0)
                if n_negative_weights > 0:
                    n_neg_weights_warnings += 1
                    if n_neg_weights_warnings <= 3:
                        logger.warning(
                            "For this value of theta, %s / %s events have negative weight and will be ignored",
                            n_negative_weights,
                            p_sampling.size,
                        )
                        if n_neg_weights_warnings == 3:
                            logger.warning("Skipping warnings about negative weights in the future...")
                    p_sampling[p_sampling < 0.0] = 0.0

                # Remove events with too large weights (not recommended)
                if n_eff_forced is not None:
                    n_too_large_weights = np.sum(p_sampling > 1.0 / n_eff_forced)
                    if n_too_large_weights > 0:
                        n_too_large_weights_warnings += 1
                        if n_too_large_weights_warnings <= 1:
                            logger.warning(
                                "For this value of theta, %s / %s events have too large weight and will be ignored",
                                n_too_large_weights,
                                p_sampling.size,
                            )
                            if n_too_large_weights_warnings == 1:
                                logger.warning("Skipping warnings about too large weights in the future...")
                        p_sampling[p_sampling > 1.0 / n_eff_forced] = 0.0

                # Remember largest weights (to calculate effective number of samples)
                largest_event_probability = max(largest_event_probability, np.max(p_sampling))

                # Calculate cumulative p (summing up all events until here)
                cumulative_p = cumulative_p.flatten()[-1] + np.cumsum(p_sampling)  # Shape: (n_batch_size,)

                # When cumulative_p hits u, we store the events
                indices = np.searchsorted(cumulative_p, u, side="left").flatten()
                # Shape: (n_samples,), values: [0, ..., n_batch_size]

                found_now = np.invert(done) & (indices < len(cumulative_p))  # Shape: (n_samples,)
                x[found_now] = x_batch[indices[found_now]]
                done[found_now] = True

                # Extract augmented data
                relevant_augmented_data = self._calculate_augmented_data(
                    augmented_data_definitions=augmented_data_definitions,
                    weights=weights[:, indices[found_now]],
                    weight_gradients=None if weight_gradients is None else weight_gradients[:, :, indices[found_now]],
                    xsecs=xsecs,
                    xsec_gradients=xsec_gradients,
                )
                for i, this_relevant_augmented_data in enumerate(relevant_augmented_data):
                    augmented_data[i][found_now] = this_relevant_augmented_data

                # Finished?
                if np.all(done):
                    break

            # Cross-check cumulative probabilities at end
            logger.debug("  Cumulative probability (should be close to 1): %s", cumulative_p[-1])

            # Check that we got 'em all, otherwise repeat
            if not np.all(done):
                logger.debug(
                    f"  After full pass through event files, {np.sum(np.invert(done))} / {done.size} "
                    f"samples not found, with u = {u[np.invert(done)]}"
                )

        n_eff_samples = 1.0 / max(1.0e-12, largest_event_probability)
        n_eff_samples = [n_eff_samples for _ in range(n_samples)]

        return (
            x,
            theta_values,
            nu_values,
            augmented_data,
            n_eff_samples,
            n_stats_warnings,
            n_neg_weights_warnings,
            n_too_large_weights_warnings,
        )

    @staticmethod
    def _calculate_augmented_data(
        augmented_data_definitions,
        weights,  # shape (n_thetas, n_events)
        weight_gradients,  # grad_theta dsigma(theta, nu) with shape (n_thetas, n_gradients, n_events)
        xsecs,  # shape (n_thetas,)
        xsec_gradients,  # grad_theta sigma(theta, nu) with shape (n_params, n_gradients)
    ):
        augmented_data = []
        for definition in augmented_data_definitions:
            if definition[0] == "ratio":
                _, i_num, i_den = definition
                ratio = (weights[i_num] / xsecs[i_num]) / (weights[i_den] / xsecs[i_den])
                ratio = ratio.reshape((-1, 1))  # (n_samples, 1)
                augmented_data.append(ratio)
            elif definition[0] == "score":
                _, i = definition
                score = weight_gradients[i, :, :] / weights[i, np.newaxis, :]  # (n_gradients, n_samples)
                score = score - xsec_gradients[i, :, np.newaxis] / xsecs[i, np.newaxis, np.newaxis]
                score = score.T  # (n_samples, n_gradients)
                augmented_data.append(score)
            else:
                raise ValueError(f"Unknown augmented data type {definition[0]}")

        return augmented_data

    def _combine_thetas_nus(self, all_thetas, all_nus):
        assert len(all_thetas) == len(all_nus)

        # all_nus is a list of a list of (None or ndarray)
        # Figure out if there's anything nontrivial in there
        add_nuisance_params = False
        for nus in all_nus:
            if self._any_nontrivial_nus(nus):
                add_nuisance_params = True

        # No nuisance params?
        if not add_nuisance_params or self.nuisance_morpher is None or self.n_nuisance_parameters == 0:
            return all_thetas

        all_combined = []
        for thetas, nus in zip(all_thetas, all_nus):
            combined = []
            if nus is None:
                nus = [None for _ in range(thetas)]
            for theta, nu in zip(thetas, nus):
                if nu is None or None in nu:
                    nu = np.zeros(self.n_nuisance_parameters)
                combined.append(np.hstack((theta, nu)))
            all_combined.append(np.asarray(combined))
        return all_combined

    @staticmethod
    def _report_effective_n_samples(all_effective_n_samples):
        if len(all_effective_n_samples) > 1:
            logger.info(
                "Effective number of samples: mean %s, with individual thetas ranging from %s to %s",
                np.mean(all_effective_n_samples),
                np.min(all_effective_n_samples),
                np.max(all_effective_n_samples),
            )
            logger.debug("Effective number of samples for all thetas: %s", all_effective_n_samples)
        else:
            logger.info("Effective number of samples: %s", all_effective_n_samples[0])

    @staticmethod
    def _parse_theta(theta, n_samples):
        theta_type_in = theta[0]
        theta_value_in = theta[1]

        if theta_type_in == "benchmark":
            thetas_out = [theta_value_in]
            if n_samples is None:
                n_samples_per_theta = 1
            else:
                n_samples_per_theta = n_samples

        elif theta_type_in == "benchmarks":
            n_benchmarks = len(theta_value_in)
            if n_samples is None:
                n_samples_per_theta = 1
            else:
                n_samples_per_theta = max(int(round(n_samples / n_benchmarks, 0)), 1)
            thetas_out = theta_value_in

        elif theta_type_in == "morphing_point":
            thetas_out = [np.asarray(theta_value_in)]
            if n_samples is None:
                n_samples_per_theta = 1
            else:
                n_samples_per_theta = n_samples

        elif theta_type_in == "morphing_points":
            n_benchmarks = len(theta_value_in)
            if n_samples is None:
                n_samples_per_theta = 1
            else:
                n_samples_per_theta = max(int(round(n_samples / n_benchmarks, 0)), 1)
            thetas_out = theta_value_in

        elif theta_type_in == "random_morphing_points":
            n_benchmarks, priors = theta_value_in
            if n_benchmarks is None or n_benchmarks <= 0 or (n_samples is not None and n_benchmarks > n_samples):
                n_benchmarks = max(n_samples, 1)
            if n_samples is None:
                n_samples_per_theta = 1
            else:
                n_samples_per_theta = max(int(round(n_samples / n_benchmarks, 0)), 1)

            thetas_out = []
            for prior in priors:
                if prior[0] == "flat":
                    prior_min = prior[1]
                    prior_max = prior[2]
                    thetas_out.append(prior_min + (prior_max - prior_min) * np.random.rand(n_benchmarks))
                elif prior[0] == "gaussian":
                    prior_mean = prior[1]
                    prior_std = prior[2]
                    thetas_out.append(np.random.normal(loc=prior_mean, scale=prior_std, size=n_benchmarks))
                else:
                    raise ValueError(f"Unknown prior {prior}")
            thetas_out = np.array(thetas_out).T

        else:
            raise ValueError(f"Unknown theta specification {theta}")

        return thetas_out, n_samples_per_theta

    def _parse_nu(self, nu, n_thetas):
        if nu is None:
            nu_type_in = "nominal"
            nu_value_in = None
        else:
            nu_type_in = nu[0]
            nu_value_in = nu[1]
        if n_thetas < 1:
            n_thetas = 1

        if nu_type_in == "nominal" or self.n_nuisance_parameters == 0:
            nu_out = [None for _ in range(n_thetas)]

        elif nu_type_in == "iid":
            priors = [nu_value_in for _ in range(self.n_nuisance_parameters)]
            return self._parse_nu(("random_morphing_points", (None, priors)), n_thetas)

        elif nu_type_in == "morphing_point":
            nu_out = np.asarray([nu_value_in for _ in range(n_thetas)])

        elif nu_type_in == "morphing_points":
            n_nus = len(nu_value_in)
            nu_out = np.asarray([nu_value_in[i % n_nus] for i in range(n_thetas)])

        elif nu_type_in == "random_morphing_points":
            _, priors = nu_value_in

            nu_out = []
            for prior in priors:
                if prior[0] == "flat":
                    prior_min = prior[1]
                    prior_max = prior[2]
                    nu_out.append(prior_min + (prior_max - prior_min) * np.random.rand(n_thetas))
                elif prior[0] == "gaussian":
                    prior_mean = prior[1]
                    prior_std = prior[2]
                    nu_out.append(np.random.normal(loc=prior_mean, scale=prior_std, size=n_thetas))
                else:
                    raise ValueError(f"Unknown prior {prior}")
            nu_out = np.array(nu_out).T

        else:
            raise ValueError(f"Unknown nu specification {nu}")

        return nu_out

    @staticmethod
    def _build_sets(thetas, nus):
        assert len(thetas) == len(nus)

        n_sets = max([len(param) for param in thetas + nus])
        sets = [[] for _ in range(n_sets)]

        for (theta, nu) in zip(thetas, nus):
            n_theta_sets_before = len(theta)
            n_nu_sets_before = len(nu)

            if n_theta_sets_before <= 0 or n_nu_sets_before <= 0:
                raise RuntimeError(
                    f"Inconsistent number of sets in _build_sets: "
                    f"thetas = {thetas}, nus = {nus}, theta = {theta}, nu = {nu}"
                )

            for i_set in range(n_sets):
                sets[i_set].append((theta[i_set % n_theta_sets_before], nu[i_set % n_nu_sets_before]))

        return sets

    @staticmethod
    def _format_sampling(theta):
        if theta[0] == "benchmark":
            return str(theta[1])
        elif theta[0] == "morphing_point":
            return str(theta[1])
        elif theta[0] == "benchmarks":
            return f"{len(theta[1])} benchmarks, starting with {theta[1][:3]}"
        elif theta[0] == "morphing_points":
            return f"{len(theta[1])} morphing points, starting with {theta[1][:3]}"
        elif theta[0] == "random_morphing_points":
            prior_str = ""
            for i, (type_, arg0, arg1) in enumerate(theta[1][1]):
                prior_str += "\n"
                if type_ == "gaussian":
                    prior_str += f"  theta_{i} ~ Gaussian with mean {arg0} and std {arg1}"
                elif type_ == "flat":
                    prior_str += f"  theta_{i} ~ flat from {arg0} to {arg1}"

            if theta[1][0] is None:
                return f"Maximally many random morphing points, drawn from the following priors: {prior_str}"
            else:
                return f"{theta[1][0]} random morphing points, drawn from the following priors: {prior_str}"
