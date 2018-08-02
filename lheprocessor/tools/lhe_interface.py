from __future__ import absolute_import, division, print_function, unicode_literals
import six

import numpy as np
from collections import OrderedDict
import uproot
import skhep.math
import logging

from lheprocessor.tools.lheanalyzer import LHEAnalysis

def extract_observables_from_lhe_file(lhe_sample_file,
                                      sampling_benchmark,
                                      observables,
                                      benchmark_names
                                      ):
    # Load LHE file
    analysis = LHEAnalysis(lhe_sample_file)
    
    # Define arrays
    weights_all_events = []
    partons_all_events = []
    
    # Scan though events
    nevents = 0
    for event in analysis:
  
        # Count events
        nevents = nevents+1
        
        # Obtain weights for each event
        weights_event = OrderedDict()
        weights_event[sampling_benchmark]=event.weight
        for weightname,weight in event.rwgts.items():
            weights_event[weightname]=weight
        weights_all_events.append(weights_event)
        
        # Obtain particles for each event
        partons = []
        for particle in event.particles:
            if particle.status==1:
                partons.append(particle.LorentzVector)
        partons_all_events.append(partons)

    # Get number of Weights, Events and Reshape
    n_weights = len(weights_all_events[0])
    n_events = len(weights_all_events)

    weights = []
    for benchmarkname in benchmark_names:
        key_weights = []
        for weight_event in weights_all_events:
            key_weights.append(weight_event[benchmarkname])
        weights.append(key_weights)
    weights = np.array(weights)

    #weights = OrderedDict()
    #for key , _ in weights_all_events[0].items():
    #    key_weights = []
    #    for weight_event in weights_all_events:
    #        key_weights.append(weight_event[key])
    #    weights[key] = key_weights

    # Obtain values for each observable in each event
    observable_values = OrderedDict()

    for obs_name, obs_definition in six.iteritems(observables):
        values_this_observable = []

        for event in range(n_events):
            variables = {'p': partons_all_events[event]}

            try:
                values_this_observable.append(eval(obs_definition, variables))
            except:
                values_this_observable.append(np.nan)

        values_this_observable = np.array(values_this_observable, dtype=np.float)
        observable_values[obs_name] = values_this_observable

    """
    # Check for existence of required observables
    combined_filter = None

    for obs_name, obs_required in six.iteritems(observables_required):
        if obs_required:
            this_filter = np.isfinite(observable_values[obs_name])
            n_pass = np.sum(this_filter)
            n_fail = np.sum(np.invert(this_filter))

            logging.info('Requiring existence of observable %s: %s events pass, %s events removed',
                         obs_name, n_pass, n_fail)

            if combined_filter is None:
                combined_filter = this_filter
            else:
                combined_filter = np.logical_and(combined_filter, this_filter)

    if np.sum(combined_filter) == 0:
        raise RuntimeError('No observations remainining!')

    # Apply filter
    for obs_name in observable_values:
        observable_values[obs_name] = observable_values[obs_name][combined_filter]

    weights = weights[:, combined_filter]
    """

    return observable_values, weights
