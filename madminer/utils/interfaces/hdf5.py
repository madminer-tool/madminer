import h5py
import logging

from contextlib import suppress
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
