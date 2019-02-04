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

n_trainsamples = int(sys.argv[1])
h5_file = sys.argv[2]
h5shuffle_file = '/home/data/madminer_example_shuffled.h5'

inputs_file = sys.argv[3]
with open(inputs_file) as f:
    inputs = yaml.safe_load(f)

combine_and_shuffle(
    [h5_file],
    h5shuffle_file 
)

sa = SampleAugmenter(h5shuffle_file)  #'data/madminer_example_shuffled.h5')

#priors and parameters
inputs_prior=inputs['prior']
inputs_prior_0=inputs_prior['parameter_0']
inputs_prior_1=inputs_prior['parameter_1']
method=str(inputs['method'])
print(type(method))
test_split=float(inputs['test_split'])

for i in range(n_trainsamples):
    #creates training samples

    if method in ['alice','alices','cascal','carl','rolr', 'rascal']:
        x, theta0, theta1, y, r_xz, t_xz = sa.extract_samples_train_ratio(
            theta0=random_morphing_thetas(int(inputs_prior['n_thetas']), 
                                          [(str(inputs_prior_0['prior_shape']), float(inputs_prior_0['prior_param_0']), float(inputs_prior_0['prior_param_1'])), 
                                          (str(inputs_prior_1['prior_shape']), float(inputs_prior_1['prior_param_0']), float(inputs_prior_1['prior_param_1']))]),
            theta1=constant_benchmark_theta(str(inputs['benchmark'])),
            n_samples=int(inputs['n_samples']['train']),
            test_split=test_split,
            folder='/home/data/Samples_'+str(i),
            filename='train'
        )
        print('inside the loop')

    if method in ['sally','sallino']:
        x, theta0, theta1, y, r_xz, t_xz = sa.extract_samples_train_local(
            theta=random_morphing_thetas(int(inputs_prior['n_thetas']), #! input dependent
                                          [(str(inputs_prior_0['prior_shape']), float(inputs_prior_0['prior_param_0']), float(inputs_prior_0['prior_param_1'])), 
                                          (str(inputs_prior_1['prior_shape']), float(inputs_prior_1['prior_param_0']), float(inputs_prior_1['prior_param_1']))]),
            n_samples=int(inputs['n_samples']['train']),
            test_split=test_split,
            folder='/home/data/Samples_'+str(i),
            filename='train'
        )

    if method in ['scandal']:
        x, theta0, theta1, y, r_xz, t_xz = sa.extract_samples_train_global(
            theta=random_morphing_thetas(int(inputs_prior['n_thetas']), #! input dependent
                                        [(str(inputs_prior_0['prior_shape']), float(inputs_prior_0['prior_param_0']), float(inputs_prior_0['prior_param_1'])), 
                                        (str(inputs_prior_1['prior_shape']), float(inputs_prior_1['prior_param_0']), float(inputs_prior_1['prior_param_1']))]),
            n_samples=int(inputs['n_samples']['train']),
            test_split=test_split,
            folder='/home/data/Samples_'+str(i),
            filename='train'
        )

#creates test samples 
_ = sa.extract_samples_test(
    theta=constant_benchmark_theta(str(inputs['benchmark'])),
    n_samples=int(inputs['n_samples']['test']),
    folder='/home/data/Samples',
    filename='test'
)

