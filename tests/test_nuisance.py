from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
from collections import OrderedDict

from madminer.core import MadMiner
from madminer.lhe import LHEProcessor
from madminer.fisherinformation import FisherInformation, profile_information


def theta_limit_madminer(xsec=0.001, lumi=1000000.0, effect_phys=0.1, effect_sys=0.1):
    # Set up MadMiner file
    miner = MadMiner()
    miner.add_parameter(
        lha_block="no one cares",
        lha_id=12345,
        parameter_name="theta",
        morphing_max_power=1,
        parameter_range=(-1.0, 1.0),
    )
    miner.add_benchmark({"theta": 0.0})
    miner.add_benchmark({"theta": 1.0})
    miner.set_morphing(include_existing_benchmarks=True, max_overall_power=1)
    miner.save(".data.h5")

    # Set up observations
    proc = LHEProcessor(".data.h5")
    proc.add_observable("x", "no one cares")
    proc.reference_benchmark = "benchmark_0"
    proc.nuisance_parameters = OrderedDict()
    proc.nuisance_parameters["nu"] = ("benchmark_nuisance", None)
    proc.observations = OrderedDict()
    proc.observations["x"] = np.array([1.0])
    proc.weights = OrderedDict()
    proc.weights["benchmark_0"] = np.array([xsec])
    proc.weights["benchmark_1"] = np.array([xsec * (1.0 + effect_phys)])
    proc.weights["benchmark_nuisance"] = np.array([xsec * (1.0 + effect_sys)])
    proc.save(".data2.h5")

    # Calculate Fisher information
    fisher = FisherInformation(".data2.h5")
    info, cov = fisher.calculate_fisher_information_full_truth(theta=np.array([0.0]), luminosity=lumi)
    constraint = fisher.calculate_fisher_information_nuisance_constraints()
    info = info + constraint
    profiled = profile_information(info, [0])

    # Uncertainty on theta
    theta_limit = profiled[0, 0] ** -0.5

    # Remove file
    os.remove(".data.h5")
    os.remove(".data2.h5")

    return theta_limit


def theta_limit_gaussian(n_expected_events, systematic_uncertainty, dsigma_dtheta):
    chi_squared_threshold = 1.0

    var_phys = 1.0 / n_expected_events
    var_sys = (systematic_uncertainty) ** 2
    var = var_phys + var_sys

    theta_limit = (chi_squared_threshold * var / dsigma_dtheta ** 2) ** 0.5

    return theta_limit


def test_nuisance():
    # Settings
    lumi = 100000.0
    n_expected_events = 10000
    physical_effect_size = 0.05
    systematic_effect_sizes = [0.0, 0.02, 0.05, 0.1]
    tolerance = 0.10

    # Calculate limits
    mm_limits = []
    gaussian_limits = []

    for systematic_effect_size in systematic_effect_sizes:
        mm_limits.append(
            theta_limit_madminer(
                lumi=lumi,
                xsec=n_expected_events / lumi,
                effect_phys=physical_effect_size,
                effect_sys=systematic_effect_size,
            )
        )
        gaussian_limits.append(
            theta_limit_gaussian(
                n_expected_events=n_expected_events,
                systematic_uncertainty=systematic_effect_size,
                dsigma_dtheta=physical_effect_size,
            )
        )

    mm_limits = np.asarray(mm_limits)
    gaussian_limits = np.asarray(gaussian_limits)

    relative_diffs = (mm_limits - gaussian_limits) / gaussian_limits

    # Print results
    header = "{:>5s}  {:>4s}  {:>4s}  |  {:>8s}  {:>8s}  {:>6s}".format(
        "n_exp", "phys", "sys", "MadMiner", "Gauss", "diff"
    )
    print("\n")
    print(header)
    print(len(header) * "-")

    for systematic_effect_size, mm, gauss, rel in zip(
        systematic_effect_sizes, mm_limits, gaussian_limits, relative_diffs
    ):
        print(
            "{:>5d}  {:4.2f}  {:4.2f}  |  {:8.3f}  {:8.3f}  {:6.3f}".format(
                n_expected_events, physical_effect_size, systematic_effect_size, mm, gauss, rel
            )
        )

    # Check that results make sense
    assert np.all(np.abs(relative_diffs) < tolerance)


if __name__ == "__main__":
    test_nuisance()
