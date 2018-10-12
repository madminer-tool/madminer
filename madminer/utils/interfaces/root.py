from __future__ import absolute_import, division, print_function, unicode_literals
import six

import numpy as np
from collections import OrderedDict
import uproot
from madminer.utils.particle import MadMinerParticle
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

    # Weights
    ar_weights = tree.array("Weight.Weight")

    n_weights = len(ar_weights[0])
    n_events = len(ar_weights)

    assert n_weights == len(weight_labels)

    weights = np.array(ar_weights).reshape((n_events, n_weights)).T

    # Get all particle properties
    photons_all_events = _get_particles_photons(tree)
    electrons_all_events = _get_particles_charged(tree, 'Electron', 0.000511)
    muons_all_events = _get_particles_charged(tree, 'Muon', 0.105)
    leptons_all_events = _get_particles_leptons(tree)
    jets_all_events = _get_particles_jets(tree)
    met_all_events = _get_particles_met(tree)

    # Observations
    observable_values = OrderedDict()

    for obs_name, obs_definition in six.iteritems(observables):
        values_this_observable = []

        # Loop over events
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

    # Cuts
    cut_values = []

    for cut, default_pass in zip(cuts, cuts_default_pass):
        values_this_cut = []

        # Loop over events
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


def _get_particles_charged(tree, name, mass):
    pts = tree.array(name + 'Electron.PT')
    etas = tree.array(name + 'Electron.Eta')
    phis = tree.array(name + 'Electron.Phi')
    pdgids = tree.array(name + 'Electron.PID')

    all_particles = []

    for ievent in range(len(pts)):
        event_particles = []

        for pt, eta, phi, pdgid in zip(pts[ievent], etas[ievent], phis[ievent], pdgids[ievent]):
            particle = MadMinerParticle()
            particle.setptetaphim(pt, eta, phi, mass)
            particle.set_pdgid(pdgid)
            event_particles.append(particle)

        all_particles.append(event_particles)

    return all_particles


def _get_particles_leptons(tree):
    pt_mu = tree.array('Muon.PT')
    eta_mu = tree.array('Muon.Eta')
    phi_mu = tree.array('Muon.Phi')
    pdgid_mu = tree.array('Muon.PID')
    pt_e = tree.array('Electron.PT')
    eta_e = tree.array('Electron.Eta')
    phi_e = tree.array('Electron.Phi')
    pdgid_e = tree.array('Electron.PID')

    all_particles = []

    for ievent in range(len(pt_mu)):
        event_particles = []

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
        event_pdgids = np.concatenate((
            pdgid_mu[ievent],
            pdgid_e[ievent]
        ))

        # Sort by descending pT
        order = np.argsort(-1. * event_pts, axis=None)
        event_pts = event_pts[order]
        event_etas = event_etas[order]
        event_phis = event_phis[order]

        # Create particles
        for pt, eta, phi, mass, pdgid in zip(event_pts, event_etas, event_phis, event_masses, event_pdgids):
            particle = MadMinerParticle()
            particle.setptetaphim(pt, eta, phi, mass)
            particle.set_pdgid(pdgid)
            event_particles.append(particle)

        all_particles.append(event_particles)

    return all_particles


def _get_particles_photons(tree):
    pts = tree.array('Photon.PT')
    etas = tree.array('Photon.Eta')
    phis = tree.array('Photon.Phi')
    es = tree.array('Photon.E')

    all_particles = []

    for ievent in range(len(pts)):
        event_particles = []

        for pt, eta, phi, e in zip(pts[ievent], etas[ievent], phis[ievent], es[ievent]):
            particle = MadMinerParticle()
            particle.setptetaphie(pt, eta, phi, e)
            particle.set_pdgid(22)
            event_particles.append(particle)

        all_particles.append(event_particles)

    return all_particles


def _get_particles_jets(tree):
    pts = tree.array('Jet.PT')
    etas = tree.array('Jet.Eta')
    phis = tree.array('Jet.Phi')
    es = tree.array('Jet.E')

    all_particles = []

    for ievent in range(len(pts)):
        event_particles = []

        for pt, eta, phi, e in zip(pts[ievent], etas[ievent], phis[ievent], es[ievent]):
            particle = MadMinerParticle()
            particle.setptetaphie(pt, eta, phi, e)
            particle.set_pdgid(9)
            event_particles.append(particle)

        all_particles.append(event_particles)

    return all_particles


def _get_particles_met(tree):
    mets = tree.array('MissingET.MET')
    phis = tree.array('MissingET.Phi')

    all_particles = []

    for ievent in range(len(mets)):
        event_particles = []

        for met, phi in zip(mets[ievent], phis[ievent]):
            particle = MadMinerParticle()
            particle.setptetaphim(met, 0., phi, 0.)
            particle.set_pdgid(0)
            event_particles.append(particle)

        all_particles.append(event_particles)

    return all_particles
