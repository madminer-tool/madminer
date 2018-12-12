from __future__ import absolute_import, division, print_function, unicode_literals
import six

import numpy as np
from collections import OrderedDict
import skhep.math
import os
import logging

from madminer.utils.various import call_command


def extract_weights_from_lhe_file(filename, sampling_benchmark, is_background, rescale_factor=1.0):
    """ Extracts weights from a LHE file and returns them as a dict with entries benchmark_name:values """

    # Untar Event file
    new_filename, extension = os.path.splitext(filename)
    if extension == ".gz":
        if not os.path.exists(new_filename):
            call_command("gunzip -k {}".format(filename))
        filename = new_filename

    # Load LHE file
    file = open(filename, "r")

    # Go to first event, also check if sum or avg
    is_average = False
    for line in file:
        if len(line.split()) > 2 and line.split()[1] == "=" and line.split()[2] == "nevents":
            number_events_runcard = float(line.split()[0])
        if len(line.split()) > 2 and line.split()[2] == "event_norm" and line.split()[0] == "average":
            is_average = True
        if line.strip() == "</init>":
            break

    # Rescale by nevent if average
    if is_average:
        rescale_factor = rescale_factor / number_events_runcard

    # Sampling benchmark default for is_background=True
    if is_background:
        sampling_benchmark = "default"

    # Read and process weights, event by event
    weights = None

    while True:
        end_of_file, _, this_weights = _read_lhe_event(file, sampling_benchmark)
        if end_of_file:
            break

        # First results
        if weights is None:
            weights = OrderedDict()
            for key in this_weights:
                weights[key] = [this_weights[key] * rescale_factor]

        # Following results: check consistency with previous results
        if len(weights) != len(this_weights):
            raise ValueError(
                "Number of weights in different LHE events incompatible: {} vs {}".format(
                    len(weights), len(this_weights)
                )
            )

        # Merge results with previous
        for key in weights:
            assert key in this_weights, "Weight label {} not found in LHE event".format(key)
            weights[key].append(this_weights[key] * rescale_factor)

    # Vectorize
    for key in weights:
        weights[key] = np.array(weights[key])

    return weights


def extract_observables_from_lhe_file(
    filename, sampling_benchmark, is_background, rescale_factor, observables, benchmark_names
):

    """ Extracts observables and weights from a LHE file """

    # Untar Event file
    new_filename, extension = os.path.splitext(filename)
    if extension == ".gz":
        if not os.path.exists(new_filename):
            call_command("gunzip -k {}".format(filename))
        filename = new_filename

    # Load LHE file
    file = open(filename, "r")

    # Go to first event, also check if sum or avg
    is_average = False
    for line in file:
        if len(line.split()) > 2 and line.split()[1] == "=" and line.split()[2] == "nevents":
            number_events_runcard = float(line.split()[0])
        if len(line.split()) > 2 and line.split()[2] == "event_norm" and line.split()[0] == "average":
            is_average = True
        if line.strip() == "</init>":
            break

    # Rescale by nevent if average
    if is_average:
        rescale_factor = rescale_factor / number_events_runcard

    # Sampling benchmark default for is_background=True
    if is_background:
        sampling_benchmark = "default"

    # Read events
    partons_all_events = []
    weights_all_events = []
    while True:
        end_of_file, event_partons, event_weights = _read_lhe_event(file, sampling_benchmark)
        if end_of_file:
            break
        weights_all_events.append(event_weights)
        partons_all_events.append(event_partons)

    # Rewrite weights
    weights = []
    if is_background:
        for benchmarkname in benchmark_names:
            key_weights = []
            for weight_event in weights_all_events:
                key_weights.append(weight_event["default"] * rescale_factor)
            weights.append(key_weights)
        weights = np.array(weights)
    else:
        for benchmarkname in benchmark_names:
            key_weights = []
            for weight_event in weights_all_events:
                key_weights.append(weight_event[benchmarkname] * rescale_factor)
            weights.append(key_weights)
        weights = np.array(weights)

    # Obtain values for each observable in each event
    observable_values = OrderedDict()
    n_events = len(partons_all_events)
    for obs_name, obs_definition in six.iteritems(observables):
        values_this_observable = []

        for event in range(n_events):
            variables = {"p": partons_all_events[event]}

            if isinstance(obs_definition, six.string_types):
                try:
                    values_this_observable.append(eval(obs_definition, variables))
                except Exception:
                    values_this_observable.append(np.nan)
            else:
                try:
                    values_this_observable.append(obs_definition(partons_all_events[event]))
                except RuntimeError:  # (SyntaxError, NameError, TypeError, ZeroDivisionError, IndexError, RuntimeError):
                    values_this_observable.append(np.nan)

        values_this_observable = np.array(values_this_observable, dtype=np.float)
        observable_values[obs_name] = values_this_observable

    return observable_values, weights


def _read_lhe_event(file, sampling_benchmark):
    # Initialize Weights and Momenta
    event_weights = OrderedDict()
    event_momenta = []

    # Some tags so that we know where in the event we are
    do_tag = False
    do_momenta = False
    do_reweight = False
    do_wait_for_reweight = False

    # Loop through lines in Event
    for line in file:
        # Skip empty/commented out lines
        if len(line) == 0:
            continue
        if line.split()[0] == "#":
            continue
        if line.strip() == "</LesHouchesEvents>":
            return True, event_momenta, event_weights
        if line.strip() == "<event>":
            do_tag = True
            continue

        # Read Tag -> first weight
        if do_tag:
            event_weights[sampling_benchmark] = float(line.split()[2])
            do_tag = False
            do_momenta = True
            continue

        # Read Momenta and store as 4-vector
        if do_momenta:
            if line.strip() == "</event>":
                return False, event_momenta, event_weights
            if line.strip() == "<mgrwt>":
                do_momenta = False
                do_wait_for_reweight = True
                continue
            if line.strip() == "<rwgt>":
                do_momenta = False
                do_reweight = True
                continue
            status = int(line.split()[1])
            if status == 1:
                px = float(line.split()[6])
                py = float(line.split()[7])
                pz = float(line.split()[8])
                en = float(line.split()[9])
                vec = skhep.math.vectors.LorentzVector()
                vec.setpxpypze(px, py, pz, en)
                event_momenta.append(vec)
            continue

        # Wait for reweight block
        if do_wait_for_reweight:
            if line.strip() == "<rwgt>":
                do_wait_for_reweight = False
                do_reweight = True
                continue

        # Read Reweighted weights
        if do_reweight:
            if line.strip() == "</rwgt>" or line.strip() == "</mgrwt>":
                do_reweight = False
                continue
            rwgtid = line[line.find("<") + 1 : line.find(">")].split("=")[1][1:-1]
            rwgtval = float(line[line.find(">") + 1 : line.find("<", line.find("<") + 1)])
            event_weights[rwgtid] = rwgtval
            continue

        # End of Event -> return
        if line.strip() == "</event>":
            return False, event_momenta, event_weights
