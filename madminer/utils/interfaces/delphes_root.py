from __future__ import absolute_import, division, print_function, unicode_literals
import six

import numpy as np
from collections import OrderedDict
import uproot
import os
import logging

from madminer.utils.particle import MadMinerParticle
from madminer.utils.various import math_commands

logger = logging.getLogger(__name__)


def parse_delphes_root_file(
    delphes_sample_file,
    observables,
    observables_required,
    observables_defaults,
    cuts,
    cuts_default_pass,
    weight_labels=None,
    use_generator_truth=False,
    acceptance_pt_min_e=None,
    acceptance_pt_min_mu=None,
    acceptance_pt_min_a=None,
    acceptance_pt_min_j=None,
    acceptance_eta_max_e=None,
    acceptance_eta_max_mu=None,
    acceptance_eta_max_a=None,
    acceptance_eta_max_j=None,
    delete_delphes_sample_file=False,
):
    """ Extracts observables and weights from a Delphes ROOT file """

    logger.debug("Parsing Delphes file %s", delphes_sample_file)

    if weight_labels is None:
        logger.debug("Not extracting weights")
    else:
        logger.debug("Extracting weights %s", weight_labels)

    # Delphes ROOT file
    root_file = uproot.open(str(delphes_sample_file))
    # The str() call is important when using numpy 1.16.0 and Python 2.7. In this combination of versions, a unicode
    # delphes_sample_file would lead to a crash.

    # Delphes tree
    tree = root_file["Delphes"]

    # Weights
    n_weights = 0
    weights = None
    if weight_labels is not None:
        try:
            weights = tree.array("Weight.Weight")

            n_weights = len(weights[0])
            n_events = len(weights)

            logger.debug("Found %s events, %s weights", n_events, n_weights)

            weights = np.array(weights).reshape((n_events, n_weights)).T
        except KeyError:
            raise RuntimeError(
                "Extracting weights from Delphes ROOT file failed. Please install inofficial patches"
                " for the MG-Pythia interface and Delphes, available upong request, or parse weights"
                " from the LHE file!"
            )
    else:
        n_events = _get_n_events(tree)
        logger.debug("Found %s events", n_events)

    # Get all particle properties
    if use_generator_truth:
        photons_all_events = _get_particles_truth(tree, acceptance_pt_min_a, acceptance_eta_max_a, [22])
        electrons_all_events = _get_particles_truth(tree, acceptance_pt_min_a, acceptance_eta_max_a, [11, -11])
        muons_all_events = _get_particles_truth(tree, acceptance_pt_min_a, acceptance_eta_max_a, [13, -13])
        leptons_all_events = _get_particles_truth_leptons(
            tree, acceptance_pt_min_e, acceptance_eta_max_e, acceptance_pt_min_mu, acceptance_eta_max_mu
        )
        jets_all_events = _get_particles_truth_jets(tree, acceptance_pt_min_j, acceptance_eta_max_j)
        met_all_events = _get_particles_truth_met(tree)

    else:
        photons_all_events = _get_particles_photons(tree, acceptance_pt_min_a, acceptance_eta_max_a)
        electrons_all_events = _get_particles_charged(
            tree, "Electron", 0.000511, -11, acceptance_pt_min_e, acceptance_eta_max_e
        )
        muons_all_events = _get_particles_charged(tree, "Muon", 0.105, -13, acceptance_pt_min_mu, acceptance_eta_max_mu)
        leptons_all_events = _get_particles_leptons(
            tree, acceptance_pt_min_e, acceptance_eta_max_e, acceptance_pt_min_mu, acceptance_eta_max_mu
        )
        jets_all_events = _get_particles_jets(tree, acceptance_pt_min_j, acceptance_eta_max_j)
        met_all_events = _get_particles_met(tree)

    # Prepare variables
    def get_objects(ievent):
        visible_momentum = MadMinerParticle()
        for p in (
            electrons_all_events[ievent]
            + jets_all_events[ievent]
            + muons_all_events[ievent]
            + photons_all_events[ievent]
        ):
            visible_momentum += p
        all_momentum = visible_momentum + met_all_events[ievent][0]

        objects = math_commands()
        objects.update(
            {
                "e": electrons_all_events[ievent],
                "j": jets_all_events[ievent],
                "a": photons_all_events[ievent],
                "mu": muons_all_events[ievent],
                "l": leptons_all_events[ievent],
                "met": met_all_events[ievent][0],
                "visible": visible_momentum,
                "all": all_momentum,
                "boost_to_com": lambda momentum: momentum.boost(all_momentum.boost_vector()),
            }
        )

        return objects

    # Observations
    observable_values = OrderedDict()

    for obs_name, obs_definition in six.iteritems(observables):
        values_this_observable = []

        # Loop over events
        for event in range(n_events):
            variables = get_objects(event)

            if isinstance(obs_definition, six.string_types):
                try:
                    values_this_observable.append(eval(obs_definition, variables))
                except (SyntaxError, NameError, TypeError, ZeroDivisionError, IndexError):
                    default = observables_defaults[obs_name]
                    if default is None:
                        default = np.nan
                    values_this_observable.append(default)
            else:
                try:
                    values_this_observable.append(
                        obs_definition(
                            leptons_all_events[event],
                            photons_all_events[event],
                            jets_all_events[event],
                            met_all_events[event][0],
                        )
                    )
                except RuntimeError:
                    default = observables_defaults[obs_name]
                    if default is None:
                        default = np.nan
                    values_this_observable.append(default)

        values_this_observable = np.array(values_this_observable, dtype=np.float)
        observable_values[obs_name] = values_this_observable

        logger.debug("  First 10 values for observable %s:\n%s", obs_name, values_this_observable[:10])

    # Cuts
    cut_values = []

    for cut, default_pass in zip(cuts, cuts_default_pass):
        values_this_cut = []

        # Loop over events
        for event in range(n_events):
            variables = get_objects(event)

            for obs_name in observable_values:
                variables[obs_name] = observable_values[obs_name][event]

            try:
                values_this_cut.append(eval(cut, variables))
            except (SyntaxError, NameError, TypeError, ZeroDivisionError, IndexError):
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

            logger.debug("  %s / %s events pass required observable %s", n_pass, n_pass + n_fail, obs_name)

            if combined_filter is None:
                combined_filter = this_filter
            else:
                combined_filter = np.logical_and(combined_filter, this_filter)

    # Check cuts
    for cut, values_this_cut in zip(cuts, cut_values):
        n_pass = np.sum(values_this_cut)
        n_fail = np.sum(np.invert(values_this_cut))

        logger.debug("  %s / %s events pass cut %s", n_pass, n_pass + n_fail, cut)

        if combined_filter is None:
            combined_filter = values_this_cut
        else:
            combined_filter = np.logical_and(combined_filter, values_this_cut)

    # Apply filter
    if combined_filter is not None:
        n_pass = np.sum(combined_filter)
        n_fail = np.sum(np.invert(combined_filter))

        if n_pass == 0:
            logger.warning("  No observations remainining!")

            return None, None, combined_filter

        logger.info("  %s / %s events pass everything", n_pass, n_pass + n_fail)

        for obs_name in observable_values:
            observable_values[obs_name] = observable_values[obs_name][combined_filter]

        if weights is not None:
            weights = weights[:, combined_filter]

    # Wrap weights
    if weights is None:
        weights_dict = None
    else:
        weights_dict = OrderedDict()
        for weight_label, this_weights in zip(weight_labels, weights):
            weights_dict[weight_label] = this_weights

    # Delete Delphes file
    if delete_delphes_sample_file:
        logger.debug("  Deleting %s", delphes_sample_file)
        os.remove(delphes_sample_file)

    return observable_values, weights_dict, combined_filter


def _get_n_events(tree):
    es = tree.array("Event")
    n_events = len(es)
    return n_events


def _get_particles_truth(tree, pt_min, eta_max, included_pdgids=None):
    es = tree.array("Particle.E")
    pts = tree.array("Particle.PT")
    etas = tree.array("Particle.Eta")
    phis = tree.array("Particle.Phi")
    charges = tree.array("Particle.Charge")
    pdgids = tree.array("Particle.PID")

    all_particles = []

    for ievent in range(len(pts)):
        event_particles = []

        for e, pt, eta, phi, pdgid in zip(es[ievent], pts[ievent], etas[ievent], phis[ievent], pdgids[ievent]):

            if pt_min is not None and pt < pt_min:
                continue
            if eta_max is not None and abs(eta) > eta_max:
                continue
            if (included_pdgids is not None) and (not pdgid in included_pdgids):
                continue

            particle = MadMinerParticle()
            particle.setptetaphie(pt, eta, phi, e)
            particle.set_pdgid(pdgid)
            event_particles.append(particle)

        all_particles.append(event_particles)

    return all_particles


def _get_particles_charged(tree, name, mass, pdgid_positive_charge, pt_min, eta_max):
    pts = tree.array(name + ".PT")
    etas = tree.array(name + ".Eta")
    phis = tree.array(name + ".Phi")
    charges = tree.array(name + ".Charge")

    all_particles = []

    for ievent in range(len(pts)):
        event_particles = []

        for pt, eta, phi, charge in zip(pts[ievent], etas[ievent], phis[ievent], charges[ievent]):

            if pt_min is not None and pt < pt_min:
                continue
            if eta_max is not None and abs(eta) > eta_max:
                continue

            pdgid = pdgid_positive_charge if charge >= 0.0 else -pdgid_positive_charge

            particle = MadMinerParticle()
            particle.setptetaphim(pt, eta, phi, mass)
            particle.set_pdgid(pdgid)
            event_particles.append(particle)

        all_particles.append(event_particles)

    return all_particles


def _get_particles_leptons(tree, pt_min_e, eta_max_e, pt_min_mu, eta_max_mu):
    pt_mu = tree.array("Muon.PT")
    eta_mu = tree.array("Muon.Eta")
    phi_mu = tree.array("Muon.Phi")
    charge_mu = tree.array("Muon.Charge")
    pt_e = tree.array("Electron.PT")
    eta_e = tree.array("Electron.Eta")
    phi_e = tree.array("Electron.Phi")
    charge_e = tree.array("Electron.Charge")

    all_particles = []

    for ievent in range(len(pt_mu)):
        event_particles = []

        # Combined muons and electrons
        event_pts = np.concatenate((pt_mu[ievent], pt_e[ievent]))
        event_etas = np.concatenate((eta_mu[ievent], eta_e[ievent]))
        event_phis = np.concatenate((phi_mu[ievent], phi_e[ievent]))
        event_masses = np.concatenate((0.105 * np.ones_like(pt_mu[ievent]), 0.000511 * np.ones_like(pt_e[ievent])))
        event_charges = np.concatenate((charge_mu[ievent], charge_e[ievent]))
        event_pdgid_positive_charges = np.concatenate(
            (-13 * np.ones_like(pt_mu[ievent], dtype=np.int), -11 * np.ones_like(pt_e[ievent], dtype=np.int))
        )

        # Sort by descending pT
        order = np.argsort(-1.0 * event_pts, axis=None)
        event_pts = event_pts[order]
        event_etas = event_etas[order]
        event_phis = event_phis[order]

        # Create particles
        for pt, eta, phi, mass, charge, pdgid_positive_charge in zip(
            event_pts, event_etas, event_phis, event_masses, event_charges, event_pdgid_positive_charges
        ):

            pdgid = pdgid_positive_charge if charge >= 0.0 else -pdgid_positive_charge

            if abs(int(pdgid)) == 11:
                if pt_min_e is not None and pt < pt_min_e:
                    continue
                if eta_max_e is not None and abs(eta) > eta_max_e:
                    continue

            elif abs(int(pdgid)) == 13:
                if pt_min_mu is not None and pt < pt_min_mu:
                    continue
                if eta_max_mu is not None and abs(eta) > eta_max_mu:
                    continue

            else:
                logger.warning("Delphes ROOT file has lepton with PDG ID %s, ignoring it", pdgid)
                continue

            particle = MadMinerParticle()
            particle.setptetaphim(pt, eta, phi, mass)
            particle.set_pdgid(pdgid)
            event_particles.append(particle)

        all_particles.append(event_particles)

    return all_particles


def _get_particles_truth_leptons(tree, pt_min_e, eta_max_e, pt_min_mu, eta_max_mu):
    es = tree.array("Particle.E")
    pts = tree.array("Particle.PT")
    etas = tree.array("Particle.Eta")
    phis = tree.array("Particle.Phi")
    charges = tree.array("Particle.Charge")
    pdgids = tree.array("Particle.PID")

    all_particles = []

    for ievent in range(len(pts)):
        event_particles = []

        for e, pt, eta, phi, pdgid in zip(es[ievent], pts[ievent], etas[ievent], phis[ievent], pdgids[ievent]):
            if pdgid not in [11, 13, -11, -13]:
                continue
            if pdgid in [11, -11] and (pt_min_e is not None and pt < pt_min_e):
                continue
            if pdgid in [11, -11] and (eta_max_e is not None and abs(eta) > eta_max_e):
                continue
            if pdgid in [13, -13] and (pt_min_mu is not None and pt < pt_min_mu):
                continue
            if pdgid in [13, -13] and (eta_max_mu is not None and abs(eta) > eta_max_mu):
                continue

            particle = MadMinerParticle()
            particle.setptetaphie(pt, eta, phi, e)
            particle.set_pdgid(pdgid)
            event_particles.append(particle)

        all_particles.append(event_particles)

    return all_particles


def _get_particles_photons(tree, pt_min, eta_max):
    pts = tree.array("Photon.PT")
    etas = tree.array("Photon.Eta")
    phis = tree.array("Photon.Phi")
    es = tree.array("Photon.E")

    all_particles = []

    for ievent in range(len(pts)):
        event_particles = []

        for pt, eta, phi, e in zip(pts[ievent], etas[ievent], phis[ievent], es[ievent]):

            if pt_min is not None and pt < pt_min:
                continue
            if eta_max is not None and abs(eta) > eta_max:
                continue

            particle = MadMinerParticle()
            particle.setptetaphie(pt, eta, phi, e)
            particle.set_pdgid(22)
            event_particles.append(particle)

        all_particles.append(event_particles)

    return all_particles


def _get_particles_jets(tree, pt_min, eta_max):
    pts = tree.array("Jet.PT")
    etas = tree.array("Jet.Eta")
    phis = tree.array("Jet.Phi")
    masses = tree.array("Jet.Mass")
    try:
        tau_tags = tree.array("Jet.TauTag")
    except:
        logger.warning("Did not find tau-tag information in Delphes ROOT file.")
        tau_tags = [0 for _ in range(len(pts))]
    try:
        b_tags = tree.array("Jet.BTag")
    except:
        logger.warning("Did not find b-tag information in Delphes ROOT file.")
        b_tags = [0 for _ in range(len(pts))]

    all_particles = []

    for ievent in range(len(pts)):
        event_particles = []

        for pt, eta, phi, mass, tau_tag, b_tag in zip(
            pts[ievent], etas[ievent], phis[ievent], masses[ievent], tau_tags[ievent], b_tags[ievent]
        ):

            if pt_min is not None and pt < pt_min:
                continue
            if eta_max is not None and abs(eta) > eta_max:
                continue

            particle = MadMinerParticle()
            particle.setptetaphim(pt, eta, phi, mass)
            particle.set_pdgid(9)
            particle.set_tags(tau_tag >= 1, b_tag >= 1, False)
            event_particles.append(particle)

        all_particles.append(event_particles)

    return all_particles


def _get_particles_truth_jets(tree, pt_min, eta_max):
    pts = tree.array("GenJet.PT")
    etas = tree.array("GenJet.Eta")
    phis = tree.array("GenJet.Phi")
    masses = tree.array("GenJet.Mass")
    try:
        tau_tags = tree.array("GenJet.TauTag")
    except:
        logger.warning("Did not find tau-tag information for GenJets in Delphes ROOT file.")
        tau_tags = [0 for _ in range(len(pts))]
    try:
        b_tags = tree.array("GenJet.BTag")
    except:
        logger.warning("Did not find b-tag information for GenJets in Delphes ROOT file.")
        b_tags = [0 for _ in range(len(pts))]

    all_particles = []

    for ievent in range(len(pts)):
        event_particles = []

        for pt, eta, phi, mass, tau_tag, b_tag in zip(
            pts[ievent], etas[ievent], phis[ievent], masses[ievent], tau_tags[ievent], b_tags[ievent]
        ):

            if pt_min is not None and pt < pt_min:
                continue
            if eta_max is not None and abs(eta) > eta_max:
                continue

            particle = MadMinerParticle()
            particle.setptetaphim(pt, eta, phi, mass)
            particle.set_pdgid(9)
            particle.set_tags(tau_tag >= 1, b_tag >= 1, False)
            event_particles.append(particle)

        all_particles.append(event_particles)

    return all_particles


def _get_particles_truth_met(tree):
    mets = tree.array("GenMissingET.MET")
    phis = tree.array("GenMissingET.Phi")

    all_particles = []

    for ievent in range(len(mets)):
        event_particles = []

        for met, phi in zip(mets[ievent], phis[ievent]):
            particle = MadMinerParticle()
            particle.setptetaphim(met, 0.0, phi, 0.0)
            particle.set_pdgid(0)
            event_particles.append(particle)

        all_particles.append(event_particles)

    return all_particles


def _get_particles_met(tree):
    mets = tree.array("MissingET.MET")
    phis = tree.array("MissingET.Phi")

    all_particles = []

    for ievent in range(len(mets)):
        event_particles = []

        for met, phi in zip(mets[ievent], phis[ievent]):
            particle = MadMinerParticle()
            particle.setptetaphim(met, 0.0, phi, 0.0)
            particle.set_pdgid(0)
            event_particles.append(particle)

        all_particles.append(event_particles)

    return all_particles
