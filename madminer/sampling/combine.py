import logging
import numpy as np

from ..utils.interfaces.madminer_hdf5 import madminer_event_loader, load_madminer_settings
from ..utils.interfaces.madminer_hdf5 import save_preformatted_events_to_madminer_file
from ..utils.interfaces.madminer_hdf5 import save_sample_summary_to_madminer_file
from ..utils.various import shuffle

logger = logging.getLogger(__name__)


def _calculate_n_events(sampling_ids, n_benchmarks):
    if sampling_ids is None:
        return None, None

    unique, counts = np.unique(sampling_ids, return_counts=True)
    results = dict(zip(unique, counts))

    n_events_backgrounds = results.get(-1, 0)
    n_events_signal_per_benchmark = np.array([results.get(i, 0) for i in range(n_benchmarks)], dtype=np.int)
    return n_events_signal_per_benchmark, n_events_backgrounds


def combine_and_shuffle(
    input_filenames,
    output_filename,
    k_factors=None,
    overwrite_existing_file=True,
    recalculate_header=True,
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

    recalculate_header : bool, optional
        Recalculates the total number of events. Default value: True.

    Returns
    -------
        None

    """

    # TODO: merge different nuisance setups

    logger.debug("Combining and shuffling samples")

    if len(input_filenames) > 1:
        logger.warning(
            "Careful: this tool assumes that all samples are generated with the same setup, including"
            " identical benchmarks (and thus morphing setup), and identical nuisance parameters. If it is used with "
            "samples with different"
            " settings, there will be wrong results! There are no explicit cross checks in place yet."
        )

    if len(input_filenames) <= 0:
        raise ValueError("Need to provide at least one input filename")

    # k factors
    if k_factors is None:
        k_factors = [1.0 for _ in input_filenames]
    elif isinstance(k_factors, float):
        k_factors = [k_factors for _ in input_filenames]

    if len(input_filenames) != len(k_factors):
        raise RuntimeError(
            f"Inconsistent length of input filenames and k factors: "
            f"{len(input_filenames)} vs {len(k_factors)}"
        )

    # Copy first file to output_filename
    logger.debug("Copying setup from %s to %s", input_filenames[0], output_filename)

    # TODO: More memory efficient strategy

    # Load events
    all_observations = None
    all_weights = None
    all_sampling_ids = None

    all_n_events_background = 0
    all_n_events_signal_per_benchmark = 0

    for i, (filename, k_factor) in enumerate(zip(input_filenames, k_factors)):
        logger.debug(
            "Loading samples from file %s / %s at %s, multiplying weights with k factor %s",
            i + 1,
            len(input_filenames),
            filename,
            k_factor,
        )

        (
            _,
            benchmarks,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            n_signal_events_generated_per_benchmark,
            n_background_events,
            _,
            _,
        ) = load_madminer_settings(filename)
        n_benchmarks = len(benchmarks)

        if n_signal_events_generated_per_benchmark is not None and n_background_events is not None:
            all_n_events_signal_per_benchmark += n_signal_events_generated_per_benchmark
            all_n_events_background += n_background_events

        for observations, weights, sampling_ids in madminer_event_loader(filename, return_sampling_ids=True):
            logger.debug("Sampling benchmarks: %s", sampling_ids)
            if all_observations is None:
                all_observations = observations
                all_weights = k_factor * weights
            else:
                all_observations = np.vstack((all_observations, observations))
                all_weights = np.vstack((all_weights, k_factor * weights))

            if all_sampling_ids is None:
                all_sampling_ids = sampling_ids
            elif sampling_ids is not None:
                all_sampling_ids = np.hstack((all_sampling_ids, sampling_ids))

        logger.debug("Combined sampling benchmarks: %s", all_sampling_ids)

    # Shuffle
    all_observations, all_weights, all_sampling_ids = shuffle(all_observations, all_weights, all_sampling_ids)

    # Recalculate header info: number of events
    if recalculate_header:
        all_n_events_signal_per_benchmark, all_n_events_background = _calculate_n_events(all_sampling_ids, n_benchmarks)

        logger.debug(
            "Recalculated event numbers per benchmark: %s, background: %s",
            all_n_events_signal_per_benchmark,
            all_n_events_background,
        )

    # Save result
    save_preformatted_events_to_madminer_file(
        filename=output_filename,
        observations=all_observations,
        weights=all_weights,
        sampling_benchmarks=all_sampling_ids,
        copy_setup_from=input_filenames[0],
        overwrite_existing_samples=overwrite_existing_file,
    )
    if all_n_events_background + np.sum(all_n_events_signal_per_benchmark) > 0:
        save_sample_summary_to_madminer_file(
            filename=output_filename,
            n_events_background=all_n_events_background,
            n_events_per_sampling_benchmark=all_n_events_signal_per_benchmark,
        )
