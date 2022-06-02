import h5py
import logging
import numpy as np
import shutil

from collections import OrderedDict
from contextlib import suppress
from typing import Callable
from typing import Iterator
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

from madminer.models import AnalysisParameter
from madminer.models import NuisanceParameter
from madminer.models import Benchmark
from madminer.models import FiniteDiffBenchmark
from madminer.models import Observable
from madminer.models import Systematic
from madminer.models import SystematicScale
from madminer.models import SystematicType
from madminer.models import SystematicValue


logger = logging.getLogger(__name__)

# Python expression that can be evaluated with the `eval` built-in without any consequences.
# Before this symbol, the empty string was used, but it is not evaluation-compatible.
# Reference: https://github.com/madminer-tool/madminer/issues/501
EMPTY_EXPR: str = str(None)


def load_madminer_settings(file_name: str, include_nuisance_benchmarks: bool) -> tuple:
    """
    Loads the complete set of Madminer settings from a HDF5 data file

    Parameters
    ----------
    file_name: str
        HDF5 file name to load the settings from
    include_nuisance_benchmarks: bool
        Whether or not to filter out the nuisance benchmarks

    Returns
    -------
    analysis_params: OrderedDict
    benchmarks: OrderedDict
    benchmark_nuisance_flags: list
    morphing_components: numpy.ndarray
    morphing_matrix numpy.ndarray
    observables: OrderedDict
    num_samples: int
    systematics: OrderedDict
    ref_benchmark: str
    nuisance_params: OrderedDict
    num_signal_events: numpy.ndarray
    num_background_events: int
    fin_differences: dict
    fin_diff_epsilon: float
    """

    analysis_params = _load_analysis_params(file_name)
    nuisance_params = _load_nuisance_params(file_name)

    (
        benchmark_names,
        benchmark_values,
        benchmark_nuisance_flags,
        benchmark_reference_flags,
    ) = _load_benchmarks(file_name)

    (
        morphing_components,
        morphing_matrix,
    ) = _load_morphing(file_name)

    (
        fin_diff_base_benchmark,
        fin_diff_shift_benchmark,
        fin_diff_epsilon,
    ) = _load_finite_diffs(file_name)

    (
        observable_names,
        observable_defs,
    ) = _load_observables(file_name)

    (
        num_signal_events,
        num_background_events,
    ) = _load_samples_summary(file_name)

    (
        sample_observations,
        _,
        _,
    ) = _load_samples(file_name)

    (
        syst_names,
        syst_types,
        syst_values,
        syst_scales,
    ) = _load_systematics(file_name)

    # Build benchmarks dictionary
    benchmarks = OrderedDict()
    for b_name, b_matrix, b_nuisance_flag in zip(benchmark_names, benchmark_values, benchmark_nuisance_flags):

        # Filter out the nuisance benchmarks
        if include_nuisance_benchmarks is False and b_nuisance_flag is True:
            continue

        benchmarks[b_name] = Benchmark.from_params(b_name, analysis_params.keys(), b_matrix)

    # Build observables dictionary
    observables = OrderedDict()
    for o_name, o_def in zip(observable_names, observable_defs):
        observables[o_name] = Observable(o_name, o_def)

    # Build systematics dictionary
    systematics = OrderedDict()
    for s_name, s_type, s_value, s_scale in zip(syst_names, syst_types, syst_values, syst_scales):
        s_type = SystematicType.from_str(s_type)
        s_scale = SystematicScale.from_str(s_scale)
        systematics[s_name] = Systematic(s_name, s_type, s_value, s_scale)

    # Build finite differences dictionary
    fin_differences = OrderedDict()
    for base_name, matrix in zip(fin_diff_base_benchmark, fin_diff_shift_benchmark):
        fin_differences[base_name] = FiniteDiffBenchmark.from_params(base_name, analysis_params.keys(), matrix)

    # Compute concrete values
    num_samples = len(sample_observations)
    ref_benchmark = [name for name, flag in zip(benchmark_names, benchmark_reference_flags) if flag]
    ref_benchmark = ref_benchmark[0] if len(ref_benchmark) > 0 else None

    return (
        analysis_params,
        benchmarks,
        benchmark_nuisance_flags,
        morphing_components,
        morphing_matrix,
        observables,
        num_samples,
        systematics,
        ref_benchmark,
        nuisance_params,
        num_signal_events,
        num_background_events,
        fin_differences,
        fin_diff_epsilon,
    )


def save_madminer_settings(
    file_name: str,
    file_override: bool,
    parameters: Dict[str, AnalysisParameter],
    benchmarks: Dict[str, Benchmark],
    morphing_components: np.ndarray = None,
    morphing_matrix: np.ndarray = None,
    systematics: Dict[str, Systematic] = None,
    finite_differences: Dict[str, FiniteDiffBenchmark] = None,
    finite_differences_epsilon: float = None,
) -> None:
    """
    Saves the complete set of Madminer settings into a HDF5 data file

    Parameters
    ----------
    file_name: str
    file_override: bool
    parameters: OrderedDict
    benchmarks: OrderedDict
    morphing_components: numpy.ndarray
    morphing_matrix: numpy.ndarray
    systematics: OrderedDict
    finite_differences: OrderedDict
    finite_differences_epsilon: float

    Returns
    -------
        None
    """

    # Unpack provided dictionaries
    benchmark_names = [b.name for b in benchmarks.values()]
    benchmark_values = [[val for val in b.values.values()] for b in benchmarks.values()]

    fin_diffs_base_benchmarks = [b.base_name for b in finite_differences.values()]
    fin_diffs_shift_benchmarks = [[shift_name for shift_name in b.shift_names.values()] for b in finite_differences.values()]

    systematics_names = [s.name for s in systematics.values()]
    systematics_types = [s.type.value for s in systematics.values()]
    systematics_scales = [s.scale.value for s in systematics.values()]
    systematics_values = [s.value for s in systematics.values()]

    # Save information within the HDF5 file
    _save_analysis_parameters(file_name, file_override, parameters)
    _save_benchmarks(file_name, file_override, benchmark_names, benchmark_values)
    _save_morphing(file_name, file_override, morphing_components, morphing_matrix)

    if len(finite_differences) > 0:
        _save_finite_diffs(
            file_name,
            file_override,
            fin_diffs_base_benchmarks,
            fin_diffs_shift_benchmarks,
            finite_differences_epsilon,
        )
    if len(systematics) > 0:
        _save_systematics(
            file_name,
            file_override,
            systematics_names,
            systematics_types,
            systematics_values,
            systematics_scales,
        )


def save_nuisance_setup(
    file_name: str,
    file_override: bool,
    nuisance_benchmarks: List[str],
    nuisance_parameters: Dict[str, NuisanceParameter],
    reference_benchmark: str,
    copy_from_path: str = None,
) -> None:
    """
    Saves the names of nuisance-defined parameters and benchmarks in a HDF5 data file

    Parameters
    ----------
    file_name: str
    file_override: bool
    nuisance_benchmarks: list
    nuisance_parameters: dict
    reference_benchmark: str
    copy_from_path: str

    Returns
    -------
        None
    """

    if copy_from_path is not None:
        with suppress(OSError):
            shutil.copyfile(copy_from_path, file_name)

    (
        benchmark_names,
        benchmark_values,
        benchmark_nuisance_flags,
        _,
    ) = _load_benchmarks(file_name)

    (
        benchmark_names,
        benchmark_values,
        benchmark_nuisance_flags,
    ) = _add_benchmarks_custom(
        benchmark_names,
        benchmark_values,
        benchmark_nuisance_flags,
        nuisance_benchmarks,
    )

    # Compute intermediate values
    benchmark_reference_flags = [name == reference_benchmark for name in benchmark_names]

    _save_benchmarks(
        file_name,
        file_override,
        benchmark_names,
        benchmark_values,
        benchmark_nuisance_flags,
        benchmark_reference_flags,
    )

    if len(nuisance_parameters) > 0:
        _save_nuisance_params(
            file_name,
            file_override,
            nuisance_parameters,
        )


def load_events(
    file_name: str,
    start_index: int = 0,
    final_index: int = None,
    batch_size: int = 100_000,
    benchmark_nuisance_flags: List[bool] = None,
    sampling_benchmark: np.ndarray = None,
    sampling_factors: np.ndarray = None,
    include_nuisance_params: bool = True,
    include_sampling_ids: bool = False,
) -> Iterator[Union[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray]
    ]
]:
    """
    Loads generated events information from a HDF5 data file

    Parameters
    ----------
    file_name: str
    start_index: int
    final_index: int
    batch_size: int
    benchmark_nuisance_flags: list
    sampling_benchmark: numpy.ndarray
    sampling_factors: numpy.ndarray
    include_nuisance_params: bool
    include_sampling_ids: bool

    Returns
    -------

    """

    # Nuisance benchmarks filtering
    if include_nuisance_params is False and benchmark_nuisance_flags is None:
        logger.warning("Lack of nuisance flags to filter out nuisance benchmarks")
        logger.warning("Processing all weights")
        include_nuisance_params = True

    elif include_nuisance_params is False and benchmark_nuisance_flags is not None:
        benchmark_filter = np.logical_not(np.array(benchmark_nuisance_flags, dtype=bool))

    (
        observations,
        weights,
        sampling_ids,
    ) = _load_samples(file_name)

    num_samples = len(observations)

    if start_index is None:
        start_index = 0
    if final_index is None:
        final_index = num_samples
    if batch_size is None:
        batch_size = num_samples

    final_index = min(num_samples, final_index)
    actual_index = start_index

    while actual_index < final_index:
        batch_final_index = min(actual_index + batch_size, final_index)

        batch_observations = observations[actual_index : batch_final_index]
        batch_weights = weights[actual_index : batch_final_index]
        batch_sampling_ids = None

        if include_nuisance_params is False:
            batch_weights = batch_weights[:, benchmark_filter]

        if sampling_ids.size > 0:
            batch_sampling_ids = sampling_ids[actual_index : batch_final_index]

            # Only return data matching sampling_benchmark
            if sampling_benchmark is not None:
                cut = np.logical_or(
                    batch_sampling_ids == sampling_benchmark,
                    batch_sampling_ids < 0,
                )

                batch_observations = batch_observations[cut]
                batch_weights = batch_weights[cut]
                batch_sampling_ids = batch_sampling_ids[cut]

            # Rescale weights based on sampling
            elif sampling_factors is not None:
                k_factors = sampling_factors[batch_sampling_ids]
                batch_weights = batch_weights * k_factors[:, np.newaxis]

        if include_sampling_ids:
            yield batch_observations, batch_weights, batch_sampling_ids
        else:
            yield batch_observations, batch_weights

        actual_index += batch_size


def save_events(
    file_name: str,
    file_override: bool,
    observables: Dict[str, Observable],
    observations: dict,
    weights: dict,
    sampling_benchmarks: List[int],
    num_signal_events: List[int],
    num_background_events: int,
) -> None:
    """
    Saves generated events information into a HDF5 data file

    Parameters
    ----------
    file_name: str
    file_override: bool
    observables: dict
    observations: dict
    weights: dict
    sampling_benchmarks: list
    num_signal_events: list
    num_background_events: int

    Returns
    -------
        None
    """

    # Unpack provided dictionaries
    observable_names = [o.name for o in observables.values()]
    observable_defs = [o.val_expression for o in observables.values()]
    observations = [val for val in observations.values()]

    _save_observables(file_name, file_override, observable_names, observable_defs)

    if weights is None or observations is None:
        return

    benchmark_names, _, _, _ = _load_benchmarks(file_name)

    logger.debug("Weight names to save in event file: %s", weights.keys())
    logger.debug("Benchmark names to save in event file: %s", benchmark_names)
    sorted_weights = _get_sorted_weights(benchmark_names, weights)

    sample_observations = np.array(observations).T
    sample_weights = np.array(sorted_weights).T
    sampling_ids = np.array(sampling_benchmarks, dtype=int)

    _save_samples(file_name, file_override, sample_observations, sample_weights, sampling_ids)
    _save_samples_summary(file_name, file_override, num_signal_events, num_background_events)


def _add_benchmarks_custom(
    benchmark_names: List[str],
    benchmark_values: List[np.ndarray],
    benchmark_nuisance_flags: List[bool],
    new_benchmark_names: List[str],
) -> Tuple[
    List[str],
    List[np.ndarray],
    List[bool]
]:
    """
    Extend the lists of benchmark properties with new custom benchmarks

    Parameters
    ----------
    benchmark_names: list
        List of benchmark names
    benchmark_values: list
        List of benchmark values per parameter
    benchmark_nuisance_flags: list
        List of flags indicating whether the benchmarks are nuisance or not
    new_benchmark_names: list
        List of custom benchmark names to add

    Returns
    -------
    benchmark_names: list
        List of benchmark names
    benchmark_values: list
        List of benchmark values per parameter
    benchmark_nuisance_flags: list
        List of flags indicating whether the benchmarks are nuisance or not
    """

    if isinstance(benchmark_values, np.ndarray):
        benchmark_values = list(benchmark_values)

    # Sort new benchmarks by name
    for new_name in sorted(new_benchmark_names):
        if new_name in benchmark_names:
            logger.debug(f"Benchmark {new_name} already in the list of benchmarks")
            continue

        logger.debug(f"Adding custom benchmark: {new_name}")
        benchmark_names.append(new_name)
        benchmark_values.append(np.zeros_like(benchmark_values[0]))
        benchmark_nuisance_flags.append(True)

    return (
        benchmark_names,
        benchmark_values,
        benchmark_nuisance_flags,
    )


def _load_benchmarks(file_name: str) -> Tuple[List[str], List[List[float]], List[bool], List[bool]]:
    """
    Load benchmark properties from a HDF5 data file.

    Parameters
    ----------
    file_name: str
        HDF5 file name to load benchmark properties from

    Returns
    -------
    benchmark_names: list
        List of benchmark names
    benchmark_values: list
        List of benchmark values per parameter
    benchmark_nuisance_flags: list
        List of flags indicating whether the benchmarks are nuisance or not
    benchmark_reference_flags: list
        List of flags indicating whether the benchmarks are reference or not
    """

    with h5py.File(file_name, "r") as file:

        # Mandatory properties
        try:
            benchmark_names = file["benchmarks/names"][()]
            benchmark_values = file["benchmarks/values"][()]
        except KeyError:
            raise IOError("Cannot read benchmarks from HDF5 file")
        else:
            benchmark_names = _decode_strings(benchmark_names)

        # Optional properties
        try:
            benchmark_nuisance_flags = file["benchmarks/is_nuisance"][()]
            benchmark_nuisance_flags = [bool(flag) for flag in benchmark_nuisance_flags]
        except KeyError:
            logger.info("HDF5 file does not contain benchmark nuisance flags")
            benchmark_nuisance_flags = [False for _ in benchmark_names]

        try:
            benchmark_reference_flags = file["benchmarks/is_reference"][()]
            benchmark_reference_flags = [bool(flag) for flag in benchmark_reference_flags]
        except KeyError:
            logger.info("HDF5 file does not contain benchmark reference flags")
            benchmark_reference_flags = [False for _ in benchmark_names]

    return (
        benchmark_names,
        benchmark_values,
        benchmark_nuisance_flags,
        benchmark_reference_flags,
    )


def _save_benchmarks(
    file_name: str,
    file_override: bool,
    benchmark_names: List[str],
    benchmark_values: List[List[float]],
    benchmark_nuisance_flags: List[bool] = None,
    benchmark_reference_flags: List[bool] = None,
) -> None:
    """
    Save benchmark properties into a HDF5 data file.

    Parameters
    ----------
    file_name: str
        HDF5 file name to save benchmark properties into
    file_override: bool
        Whether to override HDF5 file contents or not
    benchmark_names: list
        List of benchmark names
    benchmark_values: list
        List of benchmark values per parameter
    benchmark_nuisance_flags: list
        List of flags indicating whether the benchmarks are nuisance or not
    benchmark_reference_flags: list
        List of flags indicating whether the benchmarks are reference or not

    Returns
    -------
        None
    """

    if benchmark_nuisance_flags is None:
        benchmark_nuisance_flags = [False for _ in benchmark_names]

    if benchmark_reference_flags is None:
        benchmark_reference_flags = [False for _ in benchmark_names]

    benchmark_names = _encode_strings(benchmark_names)
    benchmark_nuisance_flags = [int(flag) for flag in benchmark_nuisance_flags]
    benchmark_reference_flags = [int(flag) for flag in benchmark_reference_flags]

    # Append if file exists, otherwise create
    with h5py.File(file_name, "a") as file:

        if file_override:
            with suppress(KeyError):
                del file["benchmarks"]

        # Store benchmarks
        file.create_dataset("benchmarks/names", data=benchmark_names, dtype="S256")
        file.create_dataset("benchmarks/values", data=benchmark_values)
        file.create_dataset("benchmarks/is_nuisance", data=benchmark_nuisance_flags)
        file.create_dataset("benchmarks/is_reference", data=benchmark_reference_flags)


def _load_finite_diffs(file_name: str) -> Tuple[List[str], List[List[str]], float]:
    """
    Load finite differences between benchmarks from a HDF5 data file

    Parameters
    ----------
    file_name: str
        HDF5 file name to load observables properties into

    Returns
    -------
    fin_diff_base_benchmarks: list
        List of base benchmark names
    fin_diff_shift_benchmarks: list
        List of shift benchmark names, for each of the parameters
    fin_diff_epsilon: float
        Value representing the magnitude of the numerical uncertainty
    """

    fin_diff_base_benchmarks = []
    fin_diff_shift_benchmarks = []
    fin_diff_epsilon = 0.0

    with h5py.File(file_name, "r") as file:
        try:
            fin_diff_base_benchmarks = file["finite_differences/base_benchmarks"][()]
            fin_diff_shift_benchmarks = file["finite_differences/shifted_benchmarks"][()]
            fin_diff_epsilon = float(file["finite_differences/epsilon"][()])
        except KeyError:
            logger.error("HDF5 file does not contain finite difference information")
        else:
            fin_diff_base_benchmarks = _decode_strings(fin_diff_base_benchmarks)
            fin_diff_shift_benchmarks = [_decode_strings(names) for names in fin_diff_shift_benchmarks]

    return (
        fin_diff_base_benchmarks,
        fin_diff_shift_benchmarks,
        fin_diff_epsilon,
    )


def _save_finite_diffs(
    file_name: str,
    file_override: bool,
    fin_diff_base_benchmarks: List[str],
    fin_diff_shift_benchmarks: List[List[str]],
    fin_diff_epsilon: float,
) -> None:
    """
    Save finite differences between benchmarks into a HDF5 data file

    Parameters
    ----------
    file_name: str
        HDF5 file name to save finite difference benchmarks into
    file_override: bool
        Whether to override HDF5 file contents or not
    fin_diff_base_benchmarks: list
        List of base benchmark names
    fin_diff_shift_benchmarks: list
        List of shift benchmark names, for each of the parameters
    fin_diff_epsilon: float
        Value representing the magnitude of the numerical uncertainty

    Returns
    -------
        None
    """

    fin_diff_base_benchmarks = _encode_strings(fin_diff_base_benchmarks)
    fin_diff_shift_benchmarks = [_encode_strings(names) for names in fin_diff_shift_benchmarks]

    # Append if file exists, otherwise create
    with h5py.File(file_name, "a") as file:

        if file_override:
            with suppress(KeyError):
                del file["finite_differences"]

        file.create_dataset("finite_differences/base_benchmarks", data=fin_diff_base_benchmarks)
        file.create_dataset("finite_differences/shifted_benchmarks", data=fin_diff_shift_benchmarks)
        file.create_dataset("finite_differences/epsilon", data=fin_diff_epsilon)


def _load_morphing(file_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load morphing properties from a HDF5 data file

    Parameters
    ----------
    file_name : str
        HDF5 file name to load morphing properties from

    Returns
    -------
    morphing_components : numpy.ndarray
    morphing_matrix : numpy.ndarray
    """

    morphing_components = None
    morphing_matrix = None

    with h5py.File(file_name, "r") as file:
        try:
            morphing_components = file["morphing/components"][()]
            morphing_matrix = file["morphing/morphing_matrix"][()]
        except KeyError:
            logger.error("HDF5 file does not contain morphing information")
        else:
            morphing_components = np.asarray(morphing_components, dtype=int)
            morphing_matrix = np.asarray(morphing_matrix, dtype=float)

    return morphing_components, morphing_matrix


def _save_morphing(
    file_name: str,
    file_override: bool,
    morphing_components: np.ndarray,
    morphing_matrix: np.ndarray,
) -> None:
    """
    Save morphing properties into a HDF5 data file

    Parameters
    ----------
    file_name : str
    file_override : bool
    morphing_components : numpy.ndarray
    morphing_matrix : numpy.ndarray

    Returns
    -------
        None
    """

    if morphing_components is None or morphing_matrix is None:
        return

    # Append if file exists, otherwise create
    with h5py.File(file_name, "a") as file:

        if file_override:
            with suppress(KeyError):
                del file["morphing"]

        file.create_dataset("morphing/components", data=morphing_components.astype(int))
        file.create_dataset("morphing/morphing_matrix", data=morphing_matrix.astype(float))


def _load_nuisance_params(file_name: str) -> Dict[str, NuisanceParameter]:
    """
    Load nuisance parameter properties from a HDF5 data file

    Parameters
    ----------
    file_name: str
        HDF5 file name to load nuisance parameter properties from

    Returns
    -------
    parameters: dict
        Dictionary of nuisance parameter properties to each parameter name
    """

    # Replace by normal dictionary on python 3.7+
    parameters = OrderedDict()

    with h5py.File(file_name, "r") as file:
        try:
            param_names = file["nuisance_parameters/names"][()]
            param_systematics = file["nuisance_parameters/systematics"][()]
            param_benchmarks_pos = file["nuisance_parameters/benchmark_positive"][()]
            param_benchmarks_neg = file["nuisance_parameters/benchmark_negative"][()]
        except KeyError:
            logger.error("HDF5 file does not contain nuisance parameters information")
            return parameters
        else:
            param_names = _decode_strings(param_names)
            param_systematics = _decode_strings(param_systematics)
            param_benchmarks_pos = _decode_strings(param_benchmarks_pos)
            param_benchmarks_neg = _decode_strings(param_benchmarks_neg)
            param_benchmarks_pos = [None if name == EMPTY_EXPR else name for name in param_benchmarks_pos]
            param_benchmarks_neg = [None if name == EMPTY_EXPR else name for name in param_benchmarks_neg]

    for name, sys, benchmark_pos, benchmark_neg in zip(
        param_names,
        param_systematics,
        param_benchmarks_pos,
        param_benchmarks_neg,
    ):
        parameters[name] = NuisanceParameter(name, sys, benchmark_pos, benchmark_neg)

    # TODO: The dictionary has been preserved. Harmony with other loaders?
    return parameters


def _save_nuisance_params(
    file_name: str,
    file_override: bool,
    parameters: Dict[str, NuisanceParameter],
) -> None:
    """
    Save nuisance parameter properties into a HDF5 data file

    Parameters
    ----------
    file_name: str
        HDF5 file name to save nuisance parameter properties into
    file_override: bool
        Whether to override HDF5 file contents or not
    parameters: dict
        Dictionary of nuisance parameter properties to each parameter name

    Returns
    -------
        None
    """

    param_names = [p.name for p in parameters.values()]
    param_systematics = [p.systematic for p in parameters.values()]
    param_benchmarks_pos = [p.benchmark_pos if p.benchmark_pos else EMPTY_EXPR for p in parameters.values()]
    param_benchmarks_neg = [p.benchmark_neg if p.benchmark_neg else EMPTY_EXPR for p in parameters.values()]

    param_names = _encode_strings(param_names)
    param_systematics = _encode_strings(param_systematics)
    param_benchmarks_pos = _encode_strings(param_benchmarks_pos)
    param_benchmarks_neg = _encode_strings(param_benchmarks_neg)

    # Append if file exists, otherwise create
    with h5py.File(file_name, "a") as file:

        if file_override:
            with suppress(KeyError):
                del file["nuisance_parameters"]

        file.create_dataset(
            "nuisance_parameters/names",
            data=param_names,
            dtype="S256",
        )
        file.create_dataset(
            "nuisance_parameters/systematics",
            data=param_systematics,
            dtype="S256",
        )
        file.create_dataset(
            "nuisance_parameters/benchmark_positive",
            data=param_benchmarks_pos,
            dtype="S256",
        )
        file.create_dataset(
            "nuisance_parameters/benchmark_negative",
            data=param_benchmarks_neg,
            dtype="S256",
        )


def _load_analysis_params(file_name: str) -> Dict[str, AnalysisParameter]:
    """
    Load analysis parameter properties from a HDF5 data file

    Parameters
    ----------
    file_name : str
        HDF5 file name to load analysis parameter properties from

    Returns
    -------
    parameters : dict
        Dictionary of analysis parameter properties to each parameter name
    """

    # Replace by normal dictionary on python 3.7+
    parameters = OrderedDict()

    with h5py.File(file_name, "r") as file:
        try:
            param_names = file["parameters/names"][()]
            param_lha_blocks = file["parameters/lha_blocks"][()]
            param_lha_ids = file["parameters/lha_ids"][()]
            param_max_power = file["parameters/max_power"][()]
            param_val_ranges = file["parameters/val_ranges"][()]
            param_transforms = file["parameters/transforms"][()]
        except KeyError:
            raise IOError("Cannot read parameters from HDF5 file")
        else:
            param_names = _decode_strings(param_names)
            param_lha_blocks = _decode_strings(param_lha_blocks)
            param_transforms = _decode_strings(param_transforms)

    for name, block, id, max_power, range, transform in zip(
        param_names,
        param_lha_blocks,
        param_lha_ids,
        param_max_power,
        param_val_ranges,
        param_transforms,
    ):
        parameters[name] = AnalysisParameter(
            str(name),
            str(block),
            int(id),
            int(max_power),
            tuple(range),
            str(transform),
        )

    return parameters


def _save_analysis_parameters(
    file_name: str, file_override: bool, parameters: Dict[str, AnalysisParameter]
) -> None:
    """
    Save analysis parameter properties into a HDF5 data file

    Parameters
    ----------
    file_name: str
        HDF5 file name to save analysis parameter properties into
    file_override: bool
        Whether to override HDF5 file contents or not
    parameters: dict
        Dictionary of analysis parameter properties to each parameter name

    Returns
    -------
        None
    """

    param_names = [p.name for p in parameters.values()]
    param_lha_blocks = [p.lha_block for p in parameters.values()]
    param_lha_ids = [p.lha_id for p in parameters.values()]
    param_max_power = [p.max_power for p in parameters.values()]
    param_val_ranges = [p.val_range for p in parameters.values()]
    param_transforms = [p.transform for p in parameters.values()]

    param_names = _encode_strings(param_names)
    param_lha_blocks = _encode_strings(param_lha_blocks)
    param_transforms = _encode_strings(param_transforms)

    # Append if file exists, otherwise create
    with h5py.File(file_name, "a") as file:

        if file_override:
            with suppress(KeyError):
                del file["parameters"]

        file.create_dataset("parameters/names", data=param_names, dtype="S256")
        file.create_dataset("parameters/lha_blocks", data=param_lha_blocks, dtype="S256")
        file.create_dataset("parameters/lha_ids", data=param_lha_ids)
        file.create_dataset("parameters/max_power", data=param_max_power)
        file.create_dataset("parameters/val_ranges", data=param_val_ranges)
        file.create_dataset("parameters/transforms", data=param_transforms, dtype="S256")


def _load_observables(file_name: str) -> Tuple[List[str], List[str]]:
    """
    Load observable properties from a HDF5 data file

    Parameters
    ----------
    file_name: str
        HDF5 file name to load observables properties from

    Returns
    -------
    observable_names: list
        List of observable names
    observable_defs: list
        List of observable string-encoded definitions
    """

    observable_names = []
    observable_defs = []

    with h5py.File(file_name, "r") as file:

        try:
            observable_names = file["observables/names"][()]
            observable_defs = file["observables/definitions"][()]
        except KeyError:
            logger.error("HDF5 file does not contain observables information")
        else:
            observable_names = _decode_strings(observable_names)
            observable_defs = _decode_strings(observable_defs)

    return observable_names, observable_defs


def _save_observables(
    file_name: str,
    file_override: bool,
    observable_names: List[str],
    observable_defs: List[Union[str, Callable]],
) -> None:
    """
    Save observable properties into a HDF5 data file

    Parameters
    ----------
    file_name: str
        HDF5 file name to save observables properties into
    file_override: bool
        Whether to override HDF5 file contents or not
    observable_names : list
        List of observable names
    observable_defs: list
        List of observable string-encoded or callable function definitions

    Returns
    -------
        None
    """

    observable_names = _encode_strings(observable_names)

    # Filter out callable definitions when saving into HDF5 file
    observable_defs = [d if isinstance(d, str) else EMPTY_EXPR for d in observable_defs]
    observable_defs = _encode_strings(observable_defs)

    # Append if file exists, otherwise create
    with h5py.File(file_name, "a") as file:

        if file_override:
            with suppress(KeyError):
                del file["observables"]

        file.create_dataset("observables/names", data=observable_names, dtype="S256")
        file.create_dataset("observables/definitions", data=observable_defs, dtype="S256")


def _load_samples(file_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load sample properties from a HDF5 data file

    Parameters
    ----------
    file_name: str
        HDF5 file name to load sample properties from

    Returns
    -------
    sample_observations: numpy.ndarray
    sample_weights: numpy.ndarray
    sampling_ids: numpy.ndarray
    """

    sample_observations = np.asarray([])
    sample_weights = np.asarray([])
    sampling_ids = np.asarray([])

    with h5py.File(file_name, "r") as file:
        try:
            sample_observations = file["samples/observations"][()]
            sample_weights = file["samples/weights"][()]
            sampling_ids = file["samples/sampling_benchmarks"][()]
        except KeyError:
            logger.error("HDF5 file does not contain sample information")
        else:
            assert sample_observations.shape[0] == sample_weights.shape[0], \
                "The number of sample observations and sample weights do not match"

    return (
        sample_observations,
        sample_weights,
        sampling_ids,
    )


def _save_samples(
    file_name: str,
    file_override: bool,
    sample_observations: np.ndarray,
    sample_weights: np.ndarray,
    sampling_ids: np.ndarray,
) -> None:
    """
    Load sample properties into a HDF5 data file.

    Parameters
    ----------
    file_name: str
        HDF5 file name to save sample properties into
    file_override: bool
        Whether to override HDF5 file contents or not
    sample_observations: numpy.ndarray
    sample_weights: numpy.ndarray
    sampling_ids: numpy.ndarray

    Returns
    -------
        None
    """

    assert sample_observations is not None
    assert sample_weights is not None
    assert sampling_ids is not None

    # Append if file exists, otherwise create
    with h5py.File(file_name, "a") as file:

        if file_override:
            with suppress(KeyError):
                del file["samples"]

        file.create_dataset("samples/observations", data=sample_observations)
        file.create_dataset("samples/weights", data=sample_weights)
        file.create_dataset("samples/sampling_benchmarks", data=sampling_ids)


def _load_samples_summary(file_name: str) -> Tuple[np.ndarray, int]:
    """
    Load the number of signal and background events

    Parameters
    ----------
    file_name: str
        HDF5 file name to load sample numbers from

    Returns
    -------
    num_signal_events: numpy.ndarray
        List of signal events number per benchmark
    num_background_events: int
        Number of background events
    """

    num_signal_events = np.asarray([])
    num_background_events = 0

    with h5py.File(file_name, "r") as file:
        try:
            num_signal_events = file["sample_summary/signal_events"][()]
            num_background_events = file["sample_summary/background_events"][()]
        except KeyError:
            logger.error("HDF5 file does not contain sample summary information")

    return (
        num_signal_events,
        num_background_events,
    )


def _save_samples_summary(
    file_name: str,
    file_override: bool,
    num_signal_events: List[int],
    num_background_events: int,
) -> None:
    """
    Save the number of signal and background events

    Parameters
    ----------
    file_name: str
        HDF5 file name to save sample numbers into
    file_override: bool
        Whether to override HDF5 file contents or not
    num_signal_events: list
        List with the number of signal events per benchmark
    num_background_events: int
        Number of background events

    Returns
    -------
        None
    """

    # Append if file exists, otherwise create
    with h5py.File(file_name, "a") as file:

        if file_override:
            with suppress(KeyError):
                del file["sample_summary"]

        file.create_dataset("sample_summary/signal_events", data=num_signal_events)
        file.create_dataset("sample_summary/background_events", data=num_background_events)


def _load_systematics(file_name: str) -> Tuple[List[str], List[str], List[SystematicValue], List[str]]:
    """
    Load systematics properties into a HDF5 data file.

    Parameters
    ----------
    file_name: str
        HDF5 file name to load systematics properties from

    Returns
    -------
    systematics_names: list
        List of systematics names
    systematics_types: list
        List of systematics types
    systematics_values: list
        List of systematics values (str or float)
    systematics_scales: list
        List of systematics scales
    """

    systematics_names = []
    systematics_types = []
    systematics_values = []
    systematics_scales = []

    with h5py.File(file_name, "r") as file:
        try:
            systematics_names = file["systematics/names"][()]
            systematics_types = file["systematics/types"][()]
            systematics_values = file["systematics/values"][()]
            systematics_scales = file["systematics/scales"][()]
        except KeyError:
            logger.error("HDF5 file does not contain systematic information")
        else:
            systematics_names = _decode_strings(systematics_names)
            systematics_types = _decode_strings(systematics_types)
            systematics_scales = _decode_strings(systematics_scales)
            systematics_scales = [None if scale == EMPTY_EXPR else scale for scale in systematics_scales]

    return (
        systematics_names,
        systematics_types,
        systematics_values,
        systematics_scales,
    )


def _save_systematics(
    file_name: str,
    file_override: bool,
    systematics_names: List[str],
    systematics_types: List[str],
    systematics_values: List[SystematicValue],
    systematics_scales: List[str],
) -> None:
    """
    Save systematics properties into a HDF5 data file.

    Parameters
    ----------
    file_name: str
        HDF5 file name to save systematics properties into
    file_override: bool
        Whether to override HDF5 file contents or not
    systematics_names: list
        List of systematics names
    systematics_types: list
        List of systematics types
    systematics_values: list
        List of systematics values (str or float)
    systematics_scales: list
        List of systematics scales

    Returns
    -------
        None
    """

    systematics_scales = [scale if scale else EMPTY_EXPR for scale in systematics_scales]

    systematics_names = _encode_strings(systematics_names)
    systematics_types = _encode_strings(systematics_types)
    systematics_scales = _encode_strings(systematics_scales)

    # Append if file exists, otherwise create
    with h5py.File(file_name, "a") as file:

        if file_override:
            with suppress(KeyError):
                del file["systematics"]

        file.create_dataset("systematics/names", data=systematics_names, dtype="S256")
        file.create_dataset("systematics/types", data=systematics_types, dtype="S256")
        file.create_dataset("systematics/values", data=systematics_values, dtype="S256")
        file.create_dataset("systematics/scales", data=systematics_scales, dtype="S256")


def _encode_strings(strings: List[str]) -> List[bytes]:
    """
    Encodes a list of strings as bytes

    Parameters
    ----------
    strings : list
        List of any-codification strings

    Returns
    -------
    strings: list
        List of ASCII encoded bytes
    """

    return [s.encode("ascii", "ignore") for s in strings]


def _decode_strings(strings: List[bytes]) -> List[str]:
    """
    Decodes a list of bytes into a list of strings

    Parameters
    ----------
    strings : list
        List of any-codification bytes

    Returns
    -------
    strings: list
        List of ASCII decoded strings
    """

    return [s.decode("ascii") for s in strings]


def _get_sorted_weights(benchmark_names: List[str], weights: Dict[str, float]) -> List[float]:
    """
    Extracts and sorts weight values from a dictionary.
    First the benchmarks in the right order, then the remaining ones alphabetically

    Parameters
    ----------
    benchmark_names: list
        List of benchmark names
    weights: dict
        Dictionary of benchmark-to-weight values

    Returns
    -------
    sorted_weights: list
        List of sorted weights
    """

    sorted_keys = sorted(weights.keys())

    try:
        benchmark_weights = [weights[name] for name in benchmark_names]
        remaining_weights = [weights[key] for key in sorted_keys if key not in benchmark_names]
    except KeyError as error:
        logger.warning("Issue matching weight names between the HepMC file and MadMiner:")
        logger.warning(error)
        sorted_weights = [weights[key] for key in weights]
    else:
        sorted_weights = [*benchmark_weights, *remaining_weights]

    return sorted_weights
