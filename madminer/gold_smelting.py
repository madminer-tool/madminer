from collections import OrderedDict
import tempfile

from madminer.tools.morphing import MadMorpher
from madminer.tools.h5_interface import save_madminer_file
from madminer.tools.mg_interface import export_param_card, export_reweight_card, generate_mg_process, run_mg_pythia


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


class GoldSmelter:

    def __init__(self, filename):
        self.filename = filename

        # Load file

        # Normalize xsecs of benchmarks already

        raise NotImplementedError

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
