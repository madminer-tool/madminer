import logging
import numpy as np

logger = logging.getLogger(__name__)


def benchmark(benchmark_name):
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


def benchmarks(benchmark_names):
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


def morphing_point(theta):
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
    return "morphing_point", np.asarray(theta)


def morphing_points(thetas):
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
    return "morphing_points", [np.asarray(theta) for theta in thetas]


def random_morphing_points(n_thetas, priors):
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
    return "random_morphing_points", (n_thetas, priors)


def iid_nuisance_parameters(shape="gaussian", param0=0.0, param1=1.0):
    """
    Utility function to be used as input to various SampleAugmenter functions, specifying that nuisance parameters are
    fixed at their nominal values.

    Parameters
    ----------
    shape : ["flat", "gaussian"], optional
        Parameter prior shape. Default value: "gaussian".

    param0 : float, optional
        Gaussian mean or flat lower bound. Default value: 0.0.

    param1 : float, optional
        Gaussian std or flat upper bound. Default value: 1.0.

    Returns
    -------
    output : tuple
        Input to various SampleAugmenter functions.

    """
    return "iid", (shape, param0, param1)


def nominal_nuisance_parameters():
    """
    Utility function to be used as input to various SampleAugmenter functions, specifying that nuisance parameters are
    fixed at their nominal values.

    Returns
    -------
    output : tuple
        Input to various SampleAugmenter functions

    """
    return "nominal", None
