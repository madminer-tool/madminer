from __future__ import absolute_import, division, print_function, unicode_literals
import six

import numpy as np
from collections import OrderedDict
import uproot
import skhep.math
import logging


def extract_observables_from_delphes_file(delphes_sample_file,
                                          observables,
                                          observables_required,
                                          weight_labels):
    # Delphes ROOT file
    root_file = uproot.open(delphes_sample_file)

    # Delphes tree
    tree = root_file['Delphes']

    # Weight
    ar_weights = tree.array("Weight.Weight")

    n_weights = len(ar_weights[0])
    n_events = len(ar_weights)

    assert n_weights == len(weight_labels)

    weights = np.array(ar_weights).reshape((n_events, n_weights)).T

    # Get all particle properties
    photons_all_events = _get_4vectors_photons(tree)
    electrons_all_events = _get_4vectors_electrons(tree)
    muons_all_events = _get_4vectors_muons(tree)
    jets_all_events = _get_4vectors_jets(tree)
    met_all_events = _get_4vectors_met(tree)

    # Obtain values for each observable in each event
    observable_values = OrderedDict()

    for obs_name, obs_definition in six.iteritems(observables):
        values_this_observable = []

        for event in range(n_events):
            variables = {'e': electrons_all_events[event],
                         'j': jets_all_events[event],
                         'a': photons_all_events[event],
                         'mu': muons_all_events[event],
                         'met': met_all_events[event][0]}

            try:
                values_this_observable.append(eval(obs_definition, variables))
            except:
                values_this_observable.append(np.nan)

        values_this_observable = np.array(values_this_observable, dtype=np.float)
        observable_values[obs_name] = values_this_observable

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

    # Apply filter
    if combined_filter is not None:
        if np.sum(combined_filter) == 0:
            raise RuntimeError('No observations remainining!')

        for obs_name in observable_values:
            observable_values[obs_name] = observable_values[obs_name][combined_filter]

        weights = weights[:, combined_filter]

    # Wrap weights
    weights_dict = OrderedDict()
    for weight_label, this_weights in zip(weight_labels, weights):
        weights_dict[weight_label] = this_weights

    return observable_values, weights_dict


def _get_4vectors_electrons(tree):
    pt = tree.array('Electron.PT')
    eta = tree.array('Electron.Eta')
    phi = tree.array('Electron.Phi')

    array_out = []

    for ievent, sub_list in enumerate(pt):
        array_this_event = []

        for iobject, value in enumerate(sub_list):
            vec = skhep.math.vectors.LorentzVector()

            vec.setptetaphim(pt[ievent][iobject],
                             eta[ievent][iobject],
                             phi[ievent][iobject],
                             0.000511)

            array_this_event.append(vec)

        array_out.append(array_this_event)

    return array_out


def _get_4vectors_muons(tree):
    pt = tree.array('Muon.PT')
    eta = tree.array('Muon.Eta')
    phi = tree.array('Muon.Phi')

    array_out = []

    for ievent, sub_list in enumerate(pt):
        array_this_event = []

        for iobject, value in enumerate(sub_list):
            vec = skhep.math.vectors.LorentzVector()

            vec.setptetaphim(pt[ievent][iobject],
                             eta[ievent][iobject],
                             phi[ievent][iobject],
                             0.105)

            array_this_event.append(vec)

        array_out.append(array_this_event)

    return array_out


def _get_4vectors_photons(tree):
    pt = tree.array('Photon.PT')
    eta = tree.array('Photon.Eta')
    phi = tree.array('Photon.Phi')
    e = tree.array('Photon.E')

    array_out = []

    for ievent, sub_list in enumerate(pt):
        array_this_event = []

        for iobject, value in enumerate(sub_list):
            vec = skhep.math.vectors.LorentzVector()

            vec.setptetaphie(pt[ievent][iobject],
                             eta[ievent][iobject],
                             phi[ievent][iobject],
                             e[ievent][iobject])

            array_this_event.append(vec)

        array_out.append(array_this_event)

    return array_out


def _get_4vectors_jets(tree):
    pt = tree.array('Jet.PT')
    eta = tree.array('Jet.Eta')
    phi = tree.array('Jet.Phi')
    m = tree.array('Jet.Mass')

    array_out = []

    for ievent, sub_list in enumerate(pt):
        array_this_event = []

        for iobject, value in enumerate(sub_list):
            vec = skhep.math.vectors.LorentzVector()

            vec.setptetaphim(pt[ievent][iobject],
                             eta[ievent][iobject],
                             phi[ievent][iobject],
                             m[ievent][iobject])

            array_this_event.append(vec)

        array_out.append(array_this_event)

    return array_out


def _get_4vectors_met(tree):
    met = tree.array('MissingET.MET')
    phi = tree.array('MissingET.Phi')

    array_out = []

    for ievent, sub_list in enumerate(met):
        array_this_event = []

        for iobject, value in enumerate(sub_list):
            vec = skhep.math.vectors.LorentzVector()

            vec.setptetaphim(met[ievent][iobject],
                             0.,
                             phi[ievent][iobject],
                             0.)

            array_this_event.append(vec)

        array_out.append(array_this_event)

    return array_out
