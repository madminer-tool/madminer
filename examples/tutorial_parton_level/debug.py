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
from madminer.fisherinformation import FisherInformation
from madminer.ml import ScoreEstimator

if False:
    sampler = SampleAugmenter('data/madminer_example_systematics_with_data.h5')
    x, theta, t_xz, _ = sampler.sample_train_local(
        theta=sampling.benchmark('sm'),
        n_samples=1000,
    )

    estimator = ScoreEstimator(n_hidden=(20,))
    estimator.train(
        method='sally',
        x='data/samples/x_train.npy',
        t_xz='data/samples/t_xz_train.npy',
    )
    estimator.save('models/debug')

fisher = FisherInformation('data/madminer_example_systematics_with_data.h5', include_nuisance_parameters=False)

info = fisher.calculate_fisher_information_hist1d(
    theta=[0.,0.],
    observable='pt_j1',
    luminosity=3000000.,
    nbins=10
)
