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
                                          observables_defaults,
                                          cuts,
                                          cuts_default_pass,
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
    leptons_all_events = _get_4vectors_leptons(tree)
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
                         'l': leptons_all_events[event],
                         'met': met_all_events[event][0]}

            try:
                values_this_observable.append(eval(obs_definition, variables))
            except Exception:
                default = observables_defaults[obs_name]
                if default is None:
                    default = np.nan
                values_this_observable.append(default)

        values_this_observable = np.array(values_this_observable, dtype=np.float)
        observable_values[obs_name] = values_this_observable

    # Obtain values for each cut in each event
    cut_values = []

    for cut, default_pass in zip(cuts, cuts_default_pass):
        values_this_cut = []

        for event in range(n_events):
            variables = {'e': electrons_all_events[event],
                         'j': jets_all_events[event],
                         'a': photons_all_events[event],
                         'mu': muons_all_events[event],
                         'l': leptons_all_events[event],
                         'met': met_all_events[event][0]}

            for obs_name in observable_values:
                variables[obs_name] = observable_values[obs_name][event]

            try:
                values_this_cut.append(eval(cut, variables))
            except Exception:
                values_this_cut.append(default_pass)

        values_this_cut = np.array(values_this_cut, dtype=np.bool)
        cut_values.append(values_this_cut)

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

    # Check cuts
    for cut, values_this_cut in zip(cuts, cut_values):
        n_pass = np.sum(values_this_cut)
        n_fail = np.sum(np.invert(values_this_cut))

        logging.info('Cut %s: %s events pass, %s events removed',
                     cut, n_pass, n_fail)

        if combined_filter is None:
            combined_filter = values_this_cut
        else:
            combined_filter = np.logical_and(combined_filter, values_this_cut)

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


def _get_4vectors_leptons(tree):
    pt_mu = tree.array('Muon.PT')
    eta_mu = tree.array('Muon.Eta')
    phi_mu = tree.array('Muon.Phi')
    pt_e = tree.array('Electron.PT')
    eta_e = tree.array('Electron.Eta')
    phi_e = tree.array('Electron.Phi')

    array_out = []

    for ievent in range(len(pt_mu)):
        array_this_event = []

        # Combined muons and electrons
        event_pts = np.concatenate((
            pt_mu[ievent],
            pt_e[ievent]
        ))
        event_etas = np.concatenate((
            eta_mu[ievent],
            eta_e[ievent]
        ))
        event_phis = np.concatenate((
            phi_mu[ievent],
            phi_e[ievent]
        ))
        event_masses = np.concatenate((
            0.105 * np.ones_like(pt_mu[ievent]),
            0.000511 * np.ones_like(pt_e[ievent])
        ))

        # Sort by descending pT
        order = np.argsort(-1. * event_pts, axis=None)
        event_pts = event_pts[order]
        event_etas = event_etas[order]
        event_phis = event_phis[order]

        # Create LorentzVector
        for object_pt, object_eta, object_phi, object_mass in zip(event_pts, event_etas, event_phis, event_masses):
            vec = skhep.math.vectors.LorentzVector()
            vec.setptetaphim(object_pt, object_eta, object_phi, object_mass)
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
