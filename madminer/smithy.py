from collections import OrderedDict

from madminer.tools.morphing import MadMorpher
from madminer.tools.h5_interface import load_madminer_file
from madminer.tools.utils import normalize_xsecs


def constant_benchmark_theta(benchmark_name):
    raise NotImplementedError


def multiple_benchmark_thetas(benchmark_names):
    raise NotImplementedError


def constant_morphing_theta(theta):
    raise NotImplementedError


def multiple_morphing_thetas(thetas):
    raise NotImplementedError


def random_morphing_thetas(n_thetas, prior):
    raise NotImplementedError


class Smithy:

    def __init__(self, filename):
        self.madminer_filename = filename

        # Load data
        (self.parameters, self.benchmarks, self.morphing_components, self.morphing_matrix, self.observables,
         self.observations, self.weights) = load_madminer_file(filename)

        # Normalize xsecs of benchmarks
        self.p_x_benchmarks = normalize_xsecs(self.weights)


    def _normalize_xsecs(self):
        raise NotImplementedError

    def extract_samples_local(self,
                              theta,
                              n_samples_train,
                              n_samples_test,
                              folder,
                              filename):
        """
        Extracts samples for SALLY and SALLINO

        Sampling: according to theta.
        Data: theta, x, t(x,z)
        """
        raise NotImplementedError

    def extract_samples_ratio(self,
                              theta0,
                              theta1,
                              n_samples_train,
                              n_samples_test,
                              folder,
                              filename):
        """
        Extracts samples for CARL, ROLR, CASCAL, RASCAL

        Sampling: 50% according to theta0, 50% according to theta1. theta0 can be fixed or varying, theta1 can be
        Data: theta0, theta1, x, y, r(x,z), t(x,z)
        """
        raise NotImplementedError

    # SALLY / SALLINO: need one hypothesis for sampling
