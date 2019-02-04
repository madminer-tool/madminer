from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
#import matplotlib
#from matplotlib import pyplot as plt
#%matplotlib inline
import sys 
import yaml
import inspect
from madminer.core import MadMiner
from madminer.plotting import plot_2d_morphing_basis
from madminer.delphes import DelphesProcessor
from madminer.sampling import combine_and_shuffle
from madminer.sampling import SampleAugmenter
from madminer.sampling import constant_benchmark_theta, multiple_benchmark_thetas
from madminer.sampling import constant_morphing_theta, multiple_morphing_thetas, random_morphing_thetas
from madminer.ml import MLForge


mg_dir = '/home/software/MG5_aMC_v2_6_2'

miner = MadMiner(debug=False)

input_file = sys.argv[1]

########### ADD  parameters and benchmarks from input file

with open(input_file) as f:
    # use safe_load instead load
    dict_all = yaml.safe_load(f)

#get default values of miner.add_parameters()
default_arr = inspect.getargspec(miner.add_parameter)
default = dict(zip(reversed(default_arr.args), reversed(default_arr.defaults)))

for parameter in dict_all['parameters']:
    #format range_input to tuple
    range_input = parameter['parameter_range']
    range_tuple = map(float, range_input.replace('(','').replace(')','').split(','))
   
    miner.add_parameter(
    lha_block=parameter['lha_block'], #required
    lha_id=parameter['lha_id'], #required
    parameter_name=parameter.get('parameter_name', default['parameter_name']), #optional
    morphing_max_power=int( parameter.get('morphing_max_power', default['morphing_max_power']) ), #optional
    param_card_transform=parameter.get('param_card_transform',default['param_card_transform']),  #optional
    parameter_range=range_tuple #optional
    )

for benchmark in dict_all['benchmarks']:

    miner.add_benchmark(
    {benchmark['parameter_name_1']: float(benchmark['value_1']), benchmark['parameter_name_2']:float(benchmark['value_2'])},
    benchmark['type']
    )

###########


#SET morphing
settings = dict_all['set_morphing'][0]
miner.set_morphing(
    include_existing_benchmarks=bool(settings['include_existing_benchmarks']),
    n_trials=int(settings['n_trials']),
    max_overall_power=int(settings['max_overall_power'])
)


#fig = plot_2d_morphing_basis(
#    miner.morpher,
#    xlabel=r'$c_{W} v^2 / \Lambda^2$',
#    ylabel=r'$c_{\tilde{W}} v^2 / \Lambda^2$',
#    xrange=(-10.,10.),
#    yrange=(-10.,10.)
#)

miner.save('/home/data/madminer_example.h5')
