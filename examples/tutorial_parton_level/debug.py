#! /usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
logging.basicConfig(
    format='%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s',
    datefmt='%H:%M',
    level=logging.INFO
)

from madminer import sampling
from madminer.sampling import SampleAugmenter
from madminer.fisherinformation import FisherInformation
from madminer.ml import ScoreEstimator

sampler = SampleAugmenter('data/madminer_example_systematics_with_data.h5')

x, theta0, theta1, y, r_xz, t_xz, n_effective_samples = sampler.sample_train_ratio(
    theta0=sampling.random_morphing_points(None, [('gaussian', 0., 15.), ('gaussian', 0., 15.)]),
    theta1=sampling.benchmark('sm'),
    nu0=sampling.iid_nuisance_parameters(),
    nu1=sampling.nominal_nuisance_parameters(),
    n_samples=10000,
)

logging.info("x: %s", x)
