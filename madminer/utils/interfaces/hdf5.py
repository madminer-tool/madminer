import h5py
import logging
import numpy as np

from collections import OrderedDict
from contextlib import suppress
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple


logger = logging.getLogger(__name__)


def _load_benchmarks(file_name: str) -> Tuple[List[str], List[str], List[bool], List[bool]]:
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
        List of benchmark values
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
    benchmark_values: List[str],
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
        List of benchmark values
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
            morphing_components = np.asarray(file["morphing/components"][()], dtype=np.int)
            morphing_matrix = np.asarray(file["morphing/morphing_matrix"][()], dtype=np.float)
        except KeyError:
            logger.error("HDF5 file does not contain morphing information")

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

    assert morphing_components is not None
    assert morphing_matrix is not None

    # Append if file exists, otherwise create
    with h5py.File(file_name, "a") as file:

        if file_override:
            with suppress(KeyError):
                del file["morphing"]

        file.create_dataset("morphing/components", data=morphing_components.astype(np.int))
        file.create_dataset("morphing/morphing_matrix", data=morphing_matrix.astype(np.float))


def _load_nuisance_params(file_name: str) -> Dict[str, Tuple[str, str, str]]:
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
            param_benchmarks_pos = [None if name == "" else name for name in param_benchmarks_pos]
            param_benchmarks_neg = [None if name == "" else name for name in param_benchmarks_neg]

    for name, sys, benchmark_pos, benchmark_neg in zip(
        param_names,
        param_systematics,
        param_benchmarks_pos,
        param_benchmarks_neg,
    ):
        parameters[name] = (sys, benchmark_pos, benchmark_neg)

    # TODO: The dictionary has been preserved. Harmony with other loaders?

    return parameters


def _save_nuisance_params(
    file_name: str,
    file_override: bool,
    parameters: Dict[str, Tuple[str, str, str]],
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

    param_names = _encode_strings([name for name in parameters.keys()])
    param_systematics = _encode_strings([v[0] for v in parameters.values()])

    param_benchmarks_pos = [v[1] if v[1] is not None else "" for v in parameters.values()]
    param_benchmarks_neg = [v[2] if v[2] is not None else "" for v in parameters.values()]
    param_benchmarks_pos = _encode_strings(param_benchmarks_pos)
    param_benchmarks_neg = _encode_strings(param_benchmarks_neg)

    # Append if file exists, otherwise create
    with h5py.File(file_name, "a") as file:

        if file_override:
            with suppress(KeyError):
                del file["nuisance_parameters"]

        file.create_dataset("nuisance_parameters/names", data=param_names, dtype="S256")
        file.create_dataset("nuisance_parameters/systematics", data=param_systematics, dtype="S256")
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

        # TODO: The dictionary has been preserved. Harmony with other loaders?


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
    observable_defs = [d if isinstance(d, str) else "" for d in observable_defs]
    observable_defs = _encode_strings(observable_defs)

    # Append if file exists, otherwise create
    with h5py.File(file_name, "a") as file:

        if file_override:
            with suppress(KeyError):
                del file["observables"]

        file.create_dataset("observables/names", data=observable_names, dtype="S256")
        file.create_dataset("observables/definitions", data=observable_defs, dtype="S256")


def _load_num_samples(file_name: str) -> Tuple[int, int]:
    """
    Load the number of signal and background events

    Parameters
    ----------
    file_name : str
        HDF5 file name to load sample numbers from

    Returns
    -------
    num_signal_events : int
        Number of signal generated events per benchmark
    num_background_events : int
        Number of background events
    """

    with h5py.File(file_name, "r") as file:
        try:
            num_signal_events = int(file["sample_summary/signal_events"][()])
            num_background_events = int(file["sample_summary/background_events"][()])
        except KeyError:
            num_signal_events = 0
            num_background_events = 0

    # TODO: number of samples not extracted in this function
    # TODO: return order has been swifted

    return num_signal_events, num_background_events


def _save_num_samples(
    file_name: str,
    file_override: bool,
    num_signal_events: int,
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
    num_signal_events : int
        Number of signal generated events per benchmark
    num_background_events : int
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

    # TODO: Do not check against None inputs


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
