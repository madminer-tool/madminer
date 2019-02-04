from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
#import matplotlib
#from matplotlib import pyplot as plt
#%matplotlib inline
import sys 
import yaml
from madminer.core import MadMiner
from madminer.plotting import plot_2d_morphing_basis
from madminer.delphes import DelphesProcessor
from madminer.sampling import combine_and_shuffle
from madminer.sampling import SampleAugmenter
from madminer.sampling import constant_benchmark_theta, multiple_benchmark_thetas
from madminer.sampling import constant_morphing_theta, multiple_morphing_thetas, random_morphing_thetas
from madminer.ml import MLForge

h5_file = sys.argv[1]
event_path = sys.argv[2]
input_file = sys.argv[3]
benchmark_file = sys.argv[4]

mg_dir = '/home/software/MG5_aMC_v2_6_2'

dp = DelphesProcessor(h5_file)

#### get benchmark name of the job
file = open(benchmark_file,'r')
benchmark=str(file.read())
print(benchmark)
print(type(benchmark))
####

dp.add_sample(
    lhe_filename=event_path + '/unweighted_events.lhe.gz', #'/home/code/mg_processes/signal/Events/run_01/unweighted_events.lhe.gz'
    hepmc_filename=event_path + '/tag_1_pythia8_events.hepmc.gz', #'/home/code/mg_processes/signal/Events/run_01/tag_1_pythia8_events.hepmc.gz'
    is_background=False,
    sampled_from_benchmark='sm',
    weights='lhe',
)
#benchmark,

dp.run_delphes(
    delphes_directory=mg_dir + '/Delphes',
    delphes_card='/home/code/cards/delphes_card.dat',
    log_file='/home/log_delphes.log'
    #initial_command='source activate python2'
)


########### add observables and cuts from input file
with open(input_file) as f:
    # use safe_load instead load
    dict_all = yaml.safe_load(f)

for observable in dict_all['observables']:
	dp.add_observable(
    observable['name'],
    observable['definition'],
    required=observable['required'],
    default=observable['default']
	)

for cut in dict_all['cuts']:
    dp.add_cut(cut['expression'])
#####################################

dp.analyse_delphes_samples()#reference_benchmark=benchmark

dp.save('/home/data/madminer_example_with_data.h5')
