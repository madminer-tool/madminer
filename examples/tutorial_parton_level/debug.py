#! /usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
logging.basicConfig(
    format='%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s',
    datefmt='%H:%M',
    level=logging.DEBUG
)

from madminer import sampling
from madminer.sampling import SampleAugmenter

sampler = SampleAugmenter('data/madminer_example_systematics_with_data.h5')

x, theta0, theta1, y, r_xz, t_xz, n_eff = sampler.sample_train_ratio(
    theta0=sampling.random_morphing_points(20, [('gaussian', 0., 15.), ('gaussian', 0., 15.)]),
    theta1=sampling.benchmark('sm'),
    nu0=sampling.iid_nuisance_parameters("gaussian", 0., 1.),
    nu1=sampling.nominal_nuisance_parameters(),
    n_samples=100,
)

logging.info("theta0: %s", theta0)
logging.info("theta1: %s", theta1)
