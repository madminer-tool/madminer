#! /usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import six
import logging
import numpy as np

from madminer.limits import AsymptoticLimits
from madminer.sampling import SampleAugmenter
from madminer import sampling

# MadMiner output
logging.basicConfig(
    format='%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s',
    datefmt='%H:%M',
    level=logging.INFO
)

# Output of all other modules (e.g. matplotlib)
for key in logging.Logger.manager.loggerDict:
    if "madminer" not in key:
        print("Deactivating logging output for", key)
        logging.getLogger(key).setLevel(logging.WARNING)

logging.info("Hi")

# Setup
limits = AsymptoticLimits('data/lhe_data_shuffled.h5')

theta_ranges = [(-20., 20.), (-20., 20.)]
resolutions = [25, 25]

lumi = 10000.
p_values = {}
mle = {}

# SALLY
theta_grid, p_values_expected_sally, best_fit_expected_sally = limits.expected_limits(
    mode="sally",
    model_file='models/sally',
    theta_true=[0.,0.],
    theta_ranges=theta_ranges,
    resolutions=resolutions,
    luminosity=lumi,
    include_xsec=False,
)
p_values["SALLY"] = p_values_expected_sally
mle["SALLY"] = best_fit_expected_sally

# SALLINO
theta_grid, p_values_expected_sallino, best_fit_expected_sallino = limits.expected_limits(
    mode="sallino",
    model_file='models/sally',
    theta_true=[0.,0.],
    theta_ranges=theta_ranges,
    resolutions=resolutions,
    luminosity=lumi,
    include_xsec=False,
)

p_values["SALLINO"] = p_values_expected_sallino
mle["SALLINO"] = best_fit_expected_sallino

logging.info("Done")