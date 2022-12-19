import logging
import gzip
import math
import os
import shutil
import stat

from contextlib import contextmanager
from subprocess import Popen
from subprocess import PIPE

import numpy as np

logger = logging.getLogger(__name__)


def call_command(cmd, log_file=None, return_std=False):
    if log_file is not None:
        with open(log_file, "wb") as log:
            proc = Popen(cmd, stdout=log, stderr=log, shell=True)
            _ = proc.communicate()
            exitcode = proc.returncode

        if exitcode != 0:
            raise RuntimeError(f"Calling command {cmd} returned exit code {exitcode}. Output in file {log_file}.")
    else:
        proc = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
        out, err = proc.communicate()
        exitcode = proc.returncode

        if exitcode != 0:
            raise RuntimeError(
                f"Calling command {cmd} returned exit code {exitcode}."
                f"\n\n"
                f"Std output: {out}"
                f"\n\n"
                f"Error output: {err}"
                f"\n\n"
            )

        if return_std:
            return out, err

    return exitcode


def unzip_file(filename, new_filename, block_size=65536):
    # call_command("gunzip -c {} > {}".format(filename, new_filename))  # Doesn't work under windows
    with gzip.open(filename, "rb") as s_file, open(new_filename, "wb") as d_file:
        shutil.copyfileobj(s_file, d_file, block_size)


def shuffle(*arrays):
    """Shuffles multiple arrays simultaneously"""

    permutation = None
    n_samples = None
    shuffled_arrays = []

    for i, a in enumerate(arrays):
        if a is None:
            shuffled_arrays.append(a)
            continue

        if permutation is None:
            n_samples = a.shape[0]
            permutation = np.random.permutation(n_samples)

        if a.shape[0] != n_samples:
            raise RuntimeError(
                f"Mismatching shapes when trying to simultaneously shuffle: "
                f"{[None if val is None else val.shape for val in arrays]}"
            )

        shuffled_a = a[permutation]
        shuffled_arrays.append(shuffled_a)

    return shuffled_arrays


def restrict_samplesize(n, *arrays):
    restricted_arrays = []

    for i, a in enumerate(arrays):
        if a is None:
            restricted_arrays.append(None)
            continue
        restricted_arrays.append(a[:n])

    return restricted_arrays


def balance_thetas(theta_sets_types, theta_sets_values, n_sets=None):
    """Repeats theta values such that all thetas lists have the same length"""

    if n_sets is None:
        n_sets = max([len(thetas) for thetas in theta_sets_types])

    for i, (types, values) in enumerate(zip(theta_sets_types, theta_sets_values)):
        assert len(types) == len(values)
        n_sets_before = len(types)

        if n_sets_before != n_sets:
            theta_sets_types[i] = [types[j % n_sets_before] for j in range(n_sets)]
            theta_sets_values[i] = [values[j % n_sets_before] for j in range(n_sets)]

    return theta_sets_types, theta_sets_values


def sanitize_array(array, replace_nan=0.0, replace_inf=0.0, replace_neg_inf=0.0, min_value=None, max_value=None):
    array[np.isneginf(array)] = replace_neg_inf
    array[np.isinf(array)] = replace_inf
    array[np.isnan(array)] = replace_nan

    if min_value is not None or max_value is not None:
        array = np.clip(array, min_value, max_value)

    return array


def load_and_check(filename, warning_threshold=1.0e9, memmap_files_larger_than_gb=None):
    if filename is None:
        return None

    if not isinstance(filename, str):
        data = filename
        memmap = False
    else:
        filesize_gb = os.stat(filename).st_size / 1.0 * 1024**3
        if memmap_files_larger_than_gb is None or filesize_gb <= memmap_files_larger_than_gb:
            logger.info("  Loading %s into RAM", filename)
            data = np.load(filename)
            memmap = False
        else:
            logger.info("  Loading %s as memory map", filename)
            data = np.load(filename, mmap_mode="c")
            memmap = True

    if not memmap:
        n_nans = np.sum(np.isnan(data))
        n_infs = np.sum(np.isinf(data))
        n_finite = np.sum(np.isfinite(data))
        if n_nans + n_infs > 0:
            logger.warning(
                "%s contains %s NaNs and %s Infs, compared to %s finite numbers!", filename, n_nans, n_infs, n_finite
            )

        smallest = np.nanmin(data)
        largest = np.nanmax(data)
        if np.abs(smallest) > warning_threshold or np.abs(largest) > warning_threshold:
            logger.warning(
                "Warning: file %s has some large numbers, ranging from %s to %s", filename, smallest, largest
            )

    if len(data.shape) == 1:
        data = data.reshape(-1, 1)

    return data


def math_commands():
    """Provides list with math commands - we need this when using eval"""
    return {
        "acos": math.acos,
        "asin": math.asin,
        "atan": math.atan,
        "atan2": math.atan2,
        "ceil": math.ceil,
        "cos": math.cos,
        "cosh": math.cosh,
        "exp": math.exp,
        "floor": math.floor,
        "log": math.log,
        "pi": math.pi,
        "pow": math.pow,
        "sin": math.sin,
        "sinh": math.sinh,
        "sqrt": math.sqrt,
        "tan": math.tan,
        "tanh": math.tanh,
    }


def make_file_executable(filename):
    st = os.stat(filename)
    os.chmod(filename, st.st_mode | stat.S_IEXEC)


def copy_file(source, destination):
    if source is None:
        return

    shutil.copyfile(source, destination)


def weighted_quantile(values, quantiles, sample_weight=None, values_sorted=False, old_style=False):
    """
    Calculates quantiles (similar to np.percentile), but supports weights.

    Parameters
    ----------
    values : ndarray
        Data
    quantiles : ndarray
        Which quantiles to calculate
    sample_weight : ndarray or None
        Weights
    values_sorted : bool
        If True, will avoid sorting the initial array
    old_style : bool
        If True, will correct output to be consistent with np.percentile

    Returns
    -------
    quantiles : ndarray
        Quantiles

    """

    # Input
    values = np.array(values, dtype=np.float64)
    quantiles = np.array(quantiles)

    if sample_weight is None:
        sample_weight = np.ones(len(values))

    sample_weight = np.array(sample_weight, dtype=np.float64)
    assert np.all(quantiles >= 0.0) and np.all(quantiles <= 1.0), "quantiles should be in [0, 1]"

    # Sort
    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    # Quantiles
    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight

    # Postprocessing
    if old_style:
        # To be consistent with np.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)

    return np.interp(quantiles, weighted_quantiles, values)


def approx_equal(a, b, epsilon=1.0e-6):
    return abs(a - b) < epsilon


def separate_information_blocks(fisher_information, parameters_of_interest):
    # Find indices
    n_parameters = len(fisher_information)
    n_poi = len(parameters_of_interest)

    poi_checked = []
    nuisance_params = []

    for i in range(n_parameters):
        if i in parameters_of_interest:
            poi_checked.append(i)
        else:
            nuisance_params.append(i)

    assert n_poi == len(poi_checked), "Inconsistent input"

    # Separate Fisher information parts
    information_phys = fisher_information[parameters_of_interest, :][:, parameters_of_interest]
    information_mix = fisher_information[nuisance_params, :][:, parameters_of_interest]
    information_nuisance = fisher_information[nuisance_params, :][:, nuisance_params]

    return nuisance_params, information_phys, information_mix, information_nuisance


def mdot(matrix, benchmark_information):
    """
    Calculates a product between a matrix / matrices with shape (n1) or (a, n1) and a weight list with shape (b, n2)
    or (n2,), where n1 and n2 do not have to be the same
    """

    n1 = matrix.shape[-1]
    weights_t = benchmark_information.T
    n2 = weights_t.shape[0]
    n_smaller = min(n1, n2)

    if n1 > n2:
        matrix = matrix.T
        matrix = matrix[:n_smaller]
        matrix = matrix.T
    elif n2 > n1:
        weights_t = weights_t[:n_smaller]

    return matrix.dot(weights_t)


@contextmanager
def less_logging():
    """
    Silences INFO logging messages. Based on https://gist.github.com/simon-weber/7853144
    """

    if logging.root.manager.disable != logging.DEBUG:
        yield
        return

    try:
        logging.disable(logging.INFO)
        yield
    finally:
        logging.disable(logging.DEBUG)
