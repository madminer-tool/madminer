from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import six
import os
import stat
from subprocess import Popen, PIPE
import io
import numpy as np
import shutil

from madminer import __version__

printed_splash = False


def general_init(debug=False):
    global printed_splash

    logging.basicConfig(format="%(asctime)s  %(message)s", datefmt="%H:%M")
    logging.getLogger().setLevel(logging.DEBUG if debug else logging.INFO)

    if not printed_splash:
        logging.info("")
        logging.info("------------------------------------------------------------")
        logging.info("|                                                          |")
        logging.info("|  MadMiner v{}|".format(__version__.ljust(46)))
        logging.info("|                                                          |")
        logging.info("|           Johann Brehmer, Kyle Cranmer, and Felix Kling  |")
        logging.info("|                                                          |")
        logging.info("------------------------------------------------------------")
        logging.info("")

        printed_splash = True


def call_command(cmd, log_file=None):
    logging.debug("Calling %s > %s", cmd, log_file)

    if log_file is not None:
        with io.open(log_file, "wb") as log:
            proc = Popen(cmd, stdout=log, stderr=log, shell=True)
            _ = proc.communicate()
            exitcode = proc.returncode

        if exitcode != 0:
            raise RuntimeError(
                "Calling command {} returned exit code {}. Output in file {}.".format(cmd, exitcode, log_file)
            )
    else:
        proc = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
        out, err = proc.communicate()
        exitcode = proc.returncode

        if exitcode != 0:
            raise RuntimeError(
                "Calling command {} returned exit code {}.\n\nStd output:\n\n{}Error output:\n\n{}".format(
                    cmd, exitcode, out, err
                )
            )

    return exitcode


def create_missing_folders(folders):
    if folders is None:
        return

    for folder in folders:
        if folder is None or folder == "":
            continue

        if not os.path.exists(folder):
            os.makedirs(folder)

        elif not os.path.isdir(folder):
            raise OSError("Path {} exists, but is no directory!".format(folder))


def format_benchmark(parameters, precision=2):
    output = ""

    for i, (key, value) in enumerate(six.iteritems(parameters)):
        if i > 0:
            output += ", "

        value = float(value)

        if value < 2.0 * 10.0 ** (-precision) or value > 100.0:
            output += str(key) + (" = {0:." + str(precision) + "e}").format(value)
        else:
            output += str(key) + (" = {0:." + str(precision) + "f}").format(value)

    return output


def shuffle(*arrays):
    """ Shuffles multiple arrays simultaneously """

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

        assert a.shape[0] == n_samples
        shuffled_a = a[permutation]
        shuffled_arrays.append(shuffled_a)

    return shuffled_arrays


def balance_thetas(theta_sets_types, theta_sets_values):
    """Repeats theta values such that all thetas lists have the same length """

    n_sets = max([len(thetas) for thetas in theta_sets_types])

    for i, (types, values) in enumerate(zip(theta_sets_types, theta_sets_values)):
        assert len(types) == len(values)
        n_sets_before = len(types)

        if n_sets_before != n_sets:
            theta_sets_types[i] = [types[j % n_sets_before] for j in range(n_sets)]
            theta_sets_values[i] = [values[j % n_sets_before] for j in range(n_sets)]

    return theta_sets_types, theta_sets_values


def sanitize_array(array, replace_nan=0.0, replace_inf=0.0, replace_neg_inf=0.0):
    array[np.isneginf(array)] = replace_neg_inf
    array[np.isinf(array)] = replace_inf
    array[np.isnan(array)] = replace_nan
    return array


def load_and_check(filename, warning_threshold=1.0e9):
    if filename is None:
        return None

    data = np.load(filename)

    n_nans = np.sum(np.isnan(data))
    n_infs = np.sum(np.isinf(data))
    n_finite = np.sum(np.isfinite(data))

    if n_nans + n_infs > 0:
        logging.warning(
            "Warning: file %s contains %s NaNs and %s Infs, compared to %s finite numbers!",
            filename,
            n_nans,
            n_infs,
            n_finite,
        )

    smallest = np.nanmin(data)
    largest = np.nanmax(data)

    if np.abs(smallest) > warning_threshold or np.abs(largest) > warning_threshold:
        logging.warning("Warning: file %s has some large numbers, rangin from %s to %s", filename, smallest, largest)

    return data


def math_commands():
    """Provides list with math commands - we need this when using eval"""

    from math import acos, asin, atan, atan2, ceil, cos, cosh, exp, floor, log, pi, pow, sin, sinh, sqrt, tan, tanh

    functions = [
        "acos",
        "asin",
        "atan",
        "atan2",
        "ceil",
        "cos",
        "cosh",
        "exp",
        "floor",
        "log",
        "pi",
        "pow",
        "sin",
        "sinh",
        "sqrt",
        "tan",
        "tanh",
    ]

    mathdefinitions = {}
    for function in functions:
        mathdefinitions[function] = locals().get(function, None)

    return mathdefinitions


def make_file_executable(filename):
    """

    Parameters
    ----------
    filename :
        

    Returns
    -------

    """
    st = os.stat(filename)
    os.chmod(filename, st.st_mode | stat.S_IEXEC)


def copy_file(source, destination):
    """

    Parameters
    ----------
    source :
        
    destination :
        

    Returns
    -------

    """
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
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), "quantiles should be in [0, 1]"

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
