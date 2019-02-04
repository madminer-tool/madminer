from __future__ import absolute_import, division, print_function, unicode_literals

#import logging
import numpy as np
#import matplotlib
#from matplotlib import pyplot as plt
#%matplotlib inline

import yaml
import sys

from madminer.core import MadMiner
from madminer.delphes import DelphesProcessor
from madminer.sampling import combine_and_shuffle
from madminer.sampling import SampleAugmenter
from madminer.sampling import constant_benchmark_theta, multiple_benchmark_thetas, random_morphing_thetas
from madminer.ml import MLForge
from madminer.plotting import plot_2d_morphing_basis, plot_distributions

print('init')
samples_path = str(sys.argv[1])

input_file = sys.argv[2]
with open(input_file) as f:
    # use safe_load instead load
    inputs = yaml.safe_load(f)


forge = MLForge()
print('starting to train')

forge.train(
    method=str(inputs['method']),
    theta0_filename=samples_path+'/theta0_train.npy',
    x_filename=samples_path+'/x_train.npy',
    y_filename=samples_path+'/y_train.npy',
    r_xz_filename=samples_path+'/r_xz_train.npy',
    t_xz0_filename=samples_path+'/t_xz_train.npy',
    n_hidden=(20,20),
    alpha=10.,
    n_epochs=int(inputs['n_epochs']),
    validation_split=0.3,
    batch_size=int(inputs['batch_size'])
)

print('finished to train')

forge.save('/home/models/'+str(inputs['method']))

