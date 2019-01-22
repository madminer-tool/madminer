from __future__ import absolute_import, division, print_function, unicode_literals

import six
import numpy as np
from collections import OrderedDict
import os
import logging
import xml.etree.ElementTree as ET

from madminer.utils.various import call_command, approx_equal, math_commands
from madminer.utils.particle import MadMinerParticle

logger = logging.getLogger(__name__)


def parse_lhe_file(
    filename,
    sampling_benchmark,
    observables,
    observables_required=None,
    observables_defaults=None,
    cuts=None,
    cuts_default_pass=None,
    benchmark_names=None,
    is_background=False,
    energy_resolutions=None,
    pt_resolutions=None,
    eta_resolutions=None,
    phi_resolutions=None,
    k_factor=1.0,
):
    """ Extracts observables and weights from a LHE file """

    logger.debug("Parsing LHE file %s", filename)

    # Inputs
    if k_factor is None:
        k_factor = 1.0

    if observables_required is None:
        observables_required = {key: False for key in six.iterkeys(observables)}

    if observables_defaults is None:
        observables_defaults = {key: None for key in six.iterkeys(observables)}

    if is_background and benchmark_names is None:
        raise RuntimeError("Parsing background LHE files required benchmark names to be provided.")

    if cuts is None:
        cuts = OrderedDict()

    if cuts_default_pass is None:
        cuts_default_pass = {key: False for key in six.iterkeys(cuts)}

    # Untar and open LHE file
    root = _untar_and_parse_lhe_file(filename)

    # Figure out event weighting
    run_card = root.find("header").find("MGRunCard").text

    weight_norm_is_average = None
    n_events_runcard = None
    for line in run_card.splitlines():
        # Remove run card comments
        try:
            line, _ = line.split("!")
        except:
            pass

        # Separate in keys and values
        try:
            value, key = line.split("=")
        except:
            continue

        # Remove spaces
        value = value.strip()
        key = key.strip()

        # Parse entries
        if key == "nevents":
            n_events_runcard = float(value)
        if key == "event_norm":
            weight_norm_is_average = value == "average"

            logger.debug(
                "Found entry event_norm = %s in LHE header. Interpreting this as weight_norm_is_average " "= %s.",
                value,
                weight_norm_is_average,
            )

    if weight_norm_is_average is None:
        logger.warning(
            "Cannot read weight normalization mode (entry 'event_norm') from LHE file header. MadMiner "
            "will continue assuming that events are properly normalized. Please check this!"
        )

    # If necessary, rescale by number of events
    if weight_norm_is_average is not None:
        if n_events_runcard is None:
            raise RuntimeError(
                "LHE weights have to be normalized, but MadMiner cannot read number of events (entry "
                "'nevents') from LHE file header."
            )

        k_factor = k_factor / n_events_runcard

    # Loop over events
    observations_all_events = None
    weights_all_events = None

    events = root.findall("event")

    for event in events:
        # Parse event
        particles, weights = _parse_event(event, sampling_benchmark)

        # Apply smearing
        particles = _smear_particles(particles, energy_resolutions, pt_resolutions, eta_resolutions, phi_resolutions)

        # Objects in event
        variables = _get_objects(particles)

        # Calculate observables
        observations = OrderedDict()
        for obs_name, obs_definition in six.iteritems(observables):
            if isinstance(obs_definition, six.string_types):
                try:
                    observations[obs_name] = eval(obs_definition, variables)
                except (SyntaxError, NameError, TypeError, ZeroDivisionError, IndexError):
                    if observables_required[obs_name]:
                        continue

                    default = observables_defaults[obs_name]
                    if default is None:
                        default = np.nan
                    observations[obs_name] = default
            else:
                try:
                    observations[obs_name] = obs_definition(particles)
                except RuntimeError:
                    if observables_required[obs_name]:
                        continue

                    default = observables_defaults[obs_name]
                    if default is None:
                        default = np.nan
                    observations[obs_name] = default

        # Objects for cuts
        for obs_name, obs_value in six.iteritems(observations):
            variables[obs_name] = obs_value

        # Check cuts
        for cut, default_pass in zip(cuts, cuts_default_pass):
            try:
                cut_result = eval(cut, variables)
                if not cut_result:
                    continue

            except (SyntaxError, NameError, TypeError, ZeroDivisionError, IndexError):
                if not default_pass:
                    continue

        # Reformat data
        for key, value in six.iteritems(weights):
            weights[key] = [value]
        for key, value in six.iteritems(observations):
            observations[key] = [value]

        # Store results
        if observations_all_events is None:
            observations_all_events = observations
        else:
            for key in observations_all_events:
                assert key in observations, "Observable {} not found in event".format(key)
                observations_all_events[key] = observations_all_events[key] + observations[key]

        if weights_all_events is None:
            weights_all_events = weights
        else:
            for key in weights_all_events:
                assert key in weights, "Weight {} not found in event".format(key)
                weights_all_events[key] = weights_all_events[key] + weights[key]

    # Background events
    if is_background:
        for benchmark_name in benchmark_names:
            weights_all_events[benchmark_name] = weights_all_events[sampling_benchmark]

    # k factor
    for key, value in six.iteritems(weights_all_events):
        weights_all_events[key] = k_factor * np.array(value)

    return observations_all_events, weights_all_events


def extract_nuisance_parameters_from_lhe_file(filename, systematics):
    """ Extracts the definition of nuisance parameters from the LHE file """

    logger.debug("Parsing nuisance parameter setup from LHE file at %s", filename)

    # Nuisance parameters (output)
    nuisance_params = OrderedDict()

    # When no systematics setup is defined
    if systematics is None:
        return nuisance_params

    # Parse scale factors from strings in systematics
    logger.debug("Systematics setup: %s", systematics)

    systematics_scales = []
    for key, value in six.iteritems(systematics):
        if key in ["mur", "muf", "mu"]:
            scale_factors = value.split(",")
            scale_factors = [float(sf) for sf in scale_factors]

            if len(scale_factors) == 0:
                raise RuntimeError("Cannot parse scale factor string %s", value)
            elif len(scale_factors) == 1:
                scale_factors = (scale_factors[0],)
            else:
                scale_factors = (scale_factors[-1], scale_factors[0])
            systematics_scales.append(scale_factors)
        else:
            systematics_scales.append(None)

    # Untar and parse LHE file
    root = _untar_and_parse_lhe_file(filename)

    # Find weight groups
    try:
        weight_groups = root.findall("header")[0].findall("initrwgt")[0].findall("weightgroup")
    except KeyError as e:
        raise RuntimeError("Could not find weight groups in LHE file!\n%s", e)

    if len(weight_groups) == 0:
        raise RuntimeError("Zero weight groups in LHE file!")

    # What have we already found?
    systematics_scale_done = []
    for val in systematics_scales:
        if val is None:
            systematics_scale_done.append([True, True])
        elif len(val) == 1:
            systematics_scale_done.append([False, True])
        else:
            systematics_scale_done.append([False, False])

    systematics_pdf_done = False

    # Loop over weight groups and weights and identify benchmarks
    for wg in weight_groups:
        try:
            wg_name = wg.attrib["name"]
        except KeyError:
            logger.warning("Weight group does not have name attribute")
            continue

        if "mg_reweighting" in wg_name.lower():  # Physics reweighting
            logger.debug("Found physics reweighting weight group %s", wg_name)
            continue

        elif (
            "mu" in systematics or "muf" in systematics or "mur" in systematics
        ) and "scale variation" in wg_name.lower():  # Found scale variation weight group
            logger.debug("Found scale variation weight group %s", wg_name)

            weights = wg.findall("weight")

            for weight in weights:
                try:
                    weight_id = str(weight.attrib["id"])
                    weight_muf = float(weight.attrib["MUF"])
                    weight_mur = float(weight.attrib["MUR"])
                except KeyError:
                    logger.warning("Scale variation weight does not have all expected attributes")
                    continue

                logging.debug("Found scale variation weight %s / muf = %s, mur = %s", weight_id, weight_muf, weight_mur)

                # Let's skip the entries with a varied dynamical scale for now
                try:
                    weight_dynscale = int(weight.attrib["dynscale"])
                    continue
                except KeyError:
                    pass

                # Matching time!
                for i, (syst_name, syst_scales, syst_done) in enumerate(
                    zip(systematics.keys(), systematics_scales, systematics_scale_done)
                ):
                    if syst_name == "mur":
                        for k in [0, 1]:
                            if (
                                not syst_done[k]
                                and approx_equal(weight_mur, syst_scales[k])
                                and approx_equal(weight_muf, 1.0)
                            ):
                                try:
                                    benchmarks = nuisance_params[syst_name]
                                except KeyError:
                                    benchmarks = [None, None]

                                benchmarks[k] = weight_id
                                nuisance_params[syst_name] = benchmarks

                                systematics_scale_done[i][k] = True
                                break

                    if syst_name == "muf":
                        for k in [0, 1]:
                            if (
                                not syst_done[k]
                                and approx_equal(weight_mur, 1.0)
                                and approx_equal(weight_muf, syst_scales[k])
                            ):
                                try:
                                    benchmarks = nuisance_params[syst_name]
                                except KeyError:
                                    benchmarks = [None, None]

                                benchmarks[k] = weight_id
                                nuisance_params[syst_name] = benchmarks

                                systematics_scale_done[i][k] = True
                                break

                    if syst_name == "mu":
                        for k in [0, 1]:
                            if (
                                not syst_done[k]
                                and approx_equal(weight_mur, syst_scales[k])
                                and approx_equal(weight_muf, syst_scales[k])
                            ):
                                try:
                                    benchmarks = nuisance_params[syst_name]
                                except KeyError:
                                    benchmarks = [None, None]

                                benchmarks[k] = weight_id
                                nuisance_params[syst_name] = benchmarks

                                systematics_scale_done[i][k] = True
                                break

        elif "pdf" in systematics and (
            systematics["pdf"] in wg_name.lower() or "pdf" in wg_name.lower() or "ct" in wg_name.lower()
        ):  # PDF reweighting
            logger.debug("Found PDF variation weight group %s", wg_name)

            weights = wg.findall("weight")

            for i, weight in enumerate(weights):
                try:
                    weight_id = str(weight.attrib["id"])
                    weight_pdf = int(weight.attrib["PDF"])
                except KeyError:
                    logger.warning("Scale variation weight does not have all expected attributes")
                    continue

                logger.debug("Found PDF weight %s / %s", weight_id, weight_pdf)

                # Add every PDF Hessian direction to nuisance parameters
                nuisance_params["pdf_{}".format(i)] = [weight_id, None]

                systematics_pdf_done = True

        else:
            logging.debug("Found other weight group %s", wg_name)

    # Check that everything was found
    if "pdf" in systematics.keys() and not systematics_pdf_done:
        logger.warning("Did not find benchmarks representing PDF uncertainties in LHE file!")

    for syst_name, (done1, done2) in zip(systematics.keys(), systematics_scale_done):
        if not (done1 and done2):
            logger.warning(
                "Did not find benchmarks representing scale variation uncertainty %s in LHE file!", syst_name
            )

    return nuisance_params


def _parse_event(event, sampling_benchmark):
    # Initialize weights and momenta
    weights = OrderedDict()
    particles = []

    # Split kinematics part in tag line and momenta
    event_text = event.text
    tag_line = None
    particle_lines = []
    for line in event_text.splitlines():
        elements = line.split()
        if len(elements) < 2:
            continue
        if tag_line is None:
            tag_line = elements
        else:
            particle_lines.append(elements)

    # Parse tag
    assert tag_line is not None
    weights[sampling_benchmark] = float(tag_line[2])

    # Parse momenta
    for elements in particle_lines:
        if len(elements) < 10:
            continue
        status = int(elements[1])

        if status == 1:
            pdgid = int(elements[0])
            px = float(elements[6])
            py = float(elements[7])
            pz = float(elements[8])
            e = float(elements[9])
            particle = MadMinerParticle()
            particle.setpxpypze(px, py, pz, e)
            particle.set_pdgid(pdgid)
            particles.append(particle)

    # Weights
    if event.find("rwgt") is not None:
        for weight in event.find("rwgt").findall("wgt"):
            weight_id, weight_value = weight.attrib["id"], float(weight.text)
            weights[weight_id] = weight_value

    return particles, weights


def _untar_and_parse_lhe_file(filename):
    # Untar event file
    new_filename, extension = os.path.splitext(filename)
    if extension == ".gz":
        if not os.path.exists(new_filename):
            call_command("gunzip -k {}".format(filename))
        filename = new_filename

    # In some cases, the LHE comments can contain bad characters
    with open(filename, "r") as file:
        lhe_content = file.read()
    lhe_lines = lhe_content.split("\n")
    for i, line in enumerate(lhe_lines):
        comment_pos = line.find("#")
        if comment_pos >= 0:
            lhe_lines[i] = line[:comment_pos]
    lhe_content = "\n".join(lhe_lines)

    # Parse XML tree
    root = ET.fromstring(lhe_content)

    return root


def _get_objects(particles):
    # Find visible particles
    electrons = []
    muons = []
    photons = []
    jets = []
    leptons = []
    neutrinos = []
    unstables = []
    invisibles = []

    for particle in particles:
        pdgid = abs(particle.pdgid)
        if pdgid in [1, 2, 3, 4, 5, 6, 9, 22]:
            jets.append(particle)
        elif pdgid == 11:
            electrons.append(particle)
            leptons.append(particle)
        elif pdgid == 13:
            muons.append(particle)
            leptons.append(particle)
        elif pdgid == 21:
            photons.append(particle)
        elif pdgid in [12, 14, 16]:
            neutrinos.append(particle)
            invisibles.append(particle)
        elif pdgid in [15, 23, 24, 25]:
            unstables.append(particle)
        else:
            logger.warning("Unknown particle with PDG id %s, treating as invisible!")
            invisibles.append(particle)

    # Sort by pT
    electrons = sorted(electrons, reverse=True, key=lambda x: x.pt)
    muons = sorted(muons, reverse=True, key=lambda x: x.pt)
    photons = sorted(photons, reverse=True, key=lambda x: x.pt)
    leptons = sorted(leptons, reverse=True, key=lambda x: x.pt)
    neutrinos = sorted(neutrinos, reverse=True, key=lambda x: x.pt)
    jets = sorted(jets, reverse=True, key=lambda x: x.pt)

    # MET
    met = MadMinerParticle()
    for p in invisibles:
        met += p

    # Build objects
    objects = math_commands()
    objects.update(
        {"p": particles, "e": electrons, "j": jets, "a": photons, "mu": muons, "l": leptons, "met": met, "v": neutrinos}
    )

    return objects


def _smear_variable(true_value, resolutions, id):
    """ Adds Gaussian nose to a variable """
    try:
        res = resolutions[id][0] + resolutions[id][1] * true_value

        if res <= 0.0:
            return true_value

        return true_value + np.random.normal(0.0, res, 1)

    except (TypeError, KeyError):
        return true_value


def _smear_particles(particles, energy_resolutions, pt_resolutions, eta_resolutions, phi_resolutions):
    """ Applies smearing function to particles of one event """

    # No smearing if any argument is None
    if energy_resolutions is None or pt_resolutions is None or eta_resolutions is None or phi_resolutions is None:
        return particles

    smeared_particles = []

    for particle in particles:
        pdgid = particle.pdgid

        if None in energy_resolutions[pdgid] and None in pt_resolutions[pdgid]:
            raise RuntimeError("Cannot derive both pT and energy from on-shell conditions!")

        # Smear four-momenta
        e = None
        if None not in energy_resolutions[pdgid]:
            e = -1.0
            while e < 0:
                e = _smear_variable(particle.e, energy_resolutions, pdgid)
        pt = None
        if None not in pt_resolutions[pdgid]:
            pt = -1.0
            while pt < 0:
                pt = _smear_variable(particle.pt, pt_resolutions, pdgid)
        eta = _smear_variable(particle.eta, eta_resolutions, pdgid)
        phi = _smear_variable(particle.phi(), phi_resolutions, pdgid)
        while phi > 2.0 * np.pi:
            phi -= 2.0 * np.pi
        while phi < 0.0:
            phi += 2.0 * np.pi

        # Construct particle
        smeared_particle = MadMinerParticle()

        if None in energy_resolutions[pdgid]:
            # Calculate E from on-shell conditions
            smeared_particle.setptetaphim(pt, eta, phi, particle.m)

        elif None in pt_resolutions[pdgid]:
            # Calculate pT from on-shell conditions
            if e > particle.m:
                pt = (e ** 2 - particle.m ** 2) / np.cosh(eta)
            else:
                pt = 0.0
            smeared_particle.setptetaphie(pt, eta, phi, e)

        else:
            # Everything smeared manually
            smeared_particle.setptetaphie(pt, eta, phi, e)

        # PDG id (also sets charge)
        smeared_particle.set_pdgid(pdgid)

        smeared_particles.append(smeared_particle)

    return smeared_particles
