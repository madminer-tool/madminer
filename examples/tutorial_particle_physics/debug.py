#! /usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import six
import logging
import numpy as np

from madminer.ml import MorphingAwareRatioEstimator

# MadMiner output
logging.basicConfig(
    format='%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s',
    datefmt='%H:%M',
    level=logging.DEBUG
)

# Output of all other modules (e.g. matplotlib)
for key in logging.Logger.manager.loggerDict:
    if "madminer" not in key:
        print("Deactivating logging output for", key)
        logging.getLogger(key).setLevel(logging.WARNING)

logging.info("Hi")

# Setup
ma = MorphingAwareRatioEstimator(
    "data/setup.h5",
    n_hidden=(100,),
)

ma.train(
    method="alices",
    theta="data/samples/theta0_train_ratio.npy",
    x="data/samples/x_train_ratio.npy",
    y="data/samples/y_train_ratio.npy",
    r_xz="data/samples/r_xz_train_ratio.npy",
    t_xz="data/samples/t_xz_train_ratio.npy",
    n_epochs=20,
    batch_size=100,
)

logging.info("Done")
