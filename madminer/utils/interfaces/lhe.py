import logging
import numpy as np
import xml.etree.ElementTree as ET

from collections import OrderedDict
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import List

from particle import Particle
from madminer.models import Cut
from madminer.models import Efficiency
from madminer.models import Observable
from madminer.models import Systematic
from madminer.models import SystematicScale
from madminer.models import SystematicType
from madminer.utils.particle import MadMinerParticle
from madminer.utils.various import unzip_file, approx_equal, math_commands

logger = logging.getLogger(__name__)


def parse_lhe_file(
    filename,
    sampling_benchmark,
    observables: Dict[str, Observable],
    cuts: List[Cut] = None,
    efficiencies: List[Efficiency] = None,
    benchmark_names=None,
    is_background=False,
    energy_resolutions=None,
    pt_resolutions=None,
    eta_resolutions=None,
    phi_resolutions=None,
    k_factor=1.0,
    parse_events_as_xml=True,
    systematics_dict=None,
):
    """ Extracts observables and weights from a LHE file """

    logger.debug("Parsing LHE file %s", filename)

    if parse_events_as_xml:
        logger.debug("Parsing header and events as XML with ElementTree")
    else:
        logger.debug("Parsing header as XML with ElementTree and events as text file")

    # Inputs
    if cuts is None:
        cuts = []
    if efficiencies is None:
        efficiencies = []
    if k_factor is None:
        k_factor = 1.0
    if is_background and benchmark_names is None:
        raise RuntimeError("Parsing background LHE files required benchmark names to be provided.")

    # Unzip and open LHE file
    run_card = None
    for elem in _untar_and_parse_lhe_file(filename):
        if elem.tag == "MGRunCard":
            run_card = elem.text
            break
        else:
            continue

    # Figure out event weighting
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
                "Found entry event_norm = %s in LHE header. "
                "Interpreting this as weight_norm_is_average = %s.",
                value,
                weight_norm_is_average,
            )

    if weight_norm_is_average is None:
        logger.warning(
            "Cannot read weight normalization mode (entry 'event_norm') from LHE file header. "
            "MadMiner will continue assuming that events are properly normalized. "
            "Please check this!"
        )

    # If necessary, rescale by number of events
    if weight_norm_is_average:
        if n_events_runcard is None:
            raise RuntimeError(
                "LHE weights have to be normalized, "
                "but MadMiner cannot read number of events (entry 'nevents') from LHE file header."
            )

        k_factor = k_factor / n_events_runcard

    # Loop over events
    n_events_with_negative_weights = 0
    pass_cuts = [0 for _ in cuts]
    fail_cuts = [0 for _ in cuts]
    pass_efficiencies = [0 for _ in efficiencies]
    fail_efficiencies = [0 for _ in efficiencies]
    avg_efficiencies = [0 for _ in efficiencies]

    observations_all_events = []
    weights_all_events = []
    weight_names_all_events = None

    # Option one: XML parsing
    if parse_events_as_xml:
        events = _untar_and_parse_lhe_file(filename, ["event"])
        for i_event, event in enumerate(events, start=1):
            if i_event % 100000 == 0:
                logger.info("  Processing event %d/%d", i_event, n_events_runcard)

            # Parse event
            particles, weights, global_event_data = _parse_xml_event(event, sampling_benchmark)
            n_events_with_negative_weights, observations, pass_all, weight_names_all_events, weights = _parse_event(
                avg_efficiencies,
                cuts,
                efficiencies,
                energy_resolutions,
                eta_resolutions,
                fail_cuts,
                fail_efficiencies,
                n_events_with_negative_weights,
                observables,
                particles,
                pass_cuts,
                pass_efficiencies,
                phi_resolutions,
                pt_resolutions,
                weight_names_all_events,
                weights,
                global_event_data=global_event_data,
                print_event=i_event if i_event <= 20 else 0,
            )

            # Skip events that fail anything
            if not pass_all:
                continue

            # Store results
            observations_all_events.append(observations)
            weights_all_events.append(weights)

    # Option two: text parsing
    else:
        # Iterate over events in LHE file
        for i_event, (particles, weights) in enumerate(_parse_txt_events(filename, sampling_benchmark), start=1):
            if i_event % 100000 == 0:
                logger.info("  Processing event %d/%d", i_event, n_events_runcard)

            n_events_with_negative_weights, observations, pass_all, weight_names_all_events, weights = _parse_event(
                avg_efficiencies,
                cuts,
                efficiencies,
                energy_resolutions,
                eta_resolutions,
                fail_cuts,
                fail_efficiencies,
                n_events_with_negative_weights,
                observables,
                particles,
                pass_cuts,
                pass_efficiencies,
                phi_resolutions,
                pt_resolutions,
                weight_names_all_events,
                weights,
                print_event=i_event if i_event <= 20 else 0,
            )

            # Skip events that fail anything
            if not pass_all:
                continue

            # Store results
            observations_all_events.append(observations)
            weights_all_events.append(weights)

    # Check results
    n_events_pass = _report_parse_results(
        avg_efficiencies,
        cuts,
        efficiencies,
        fail_cuts,
        fail_efficiencies,
        n_events_with_negative_weights,
        observations_all_events,
        pass_cuts,
        pass_efficiencies,
    )

    if n_events_pass == 0:
        logger.warning("  No observations remaining!")
        return None, None

    # Reformat observations to OrderedDicts with entries {observable_name : (n_events,)}
    observations_all_events = list(map(list, zip(*observations_all_events)))  # transposes to (n_observables, n_events)
    observations_dict = OrderedDict()
    for key, values in zip(observables.keys(), observations_all_events):
        observations_dict[key] = np.asarray(values)

    # Reformat weights and add k-factors to weights
    weights_all_events = np.array(weights_all_events)  # (n_events, n_weights)
    weights_all_events = k_factor * weights_all_events
    weights_all_events = OrderedDict(zip(weight_names_all_events, weights_all_events.T))

    # Background events
    if is_background:
        for benchmark_name in benchmark_names:
            weights_all_events[benchmark_name] = weights_all_events[sampling_benchmark]

    # Re-organize weights again -- necessary for background events and nuisance benchmarks
    output_weights = OrderedDict()
    for benchmark_name in benchmark_names:
        if is_background:
            output_weights[benchmark_name] = weights_all_events[sampling_benchmark]
        else:
            output_weights[benchmark_name] = weights_all_events[benchmark_name]

    for syst_name, syst_data in systematics_dict.items():
        for (
            nuisance_param_name,
            ((nuisance_benchmark0, weight_name0), (nuisance_benchmark1, weight_name1), processing),
        ) in syst_data.items():
            # Store first benchmark associated with nuisance param
            if weight_name0 is None:
                weight_name0 = sampling_benchmark
            if processing is None:
                output_weights[nuisance_benchmark0] = weights_all_events[weight_name0]
            elif isinstance(processing, float):
                output_weights[nuisance_benchmark0] = processing * weights_all_events[weight_name0]
            else:
                raise RuntimeError(f"Unknown nuisance processing {processing}")

            # Store second benchmark associated with nuisance param
            if nuisance_benchmark1 is None or weight_name1 is None:
                continue
            if processing is None:
                output_weights[nuisance_benchmark1] = weights_all_events[weight_name1]
            elif isinstance(processing, float):
                output_weights[nuisance_benchmark1] = processing * weights_all_events[weight_name1]
            else:
                raise RuntimeError(f"Unknown nuisance processing {processing}")

    return observations_dict, output_weights


def _report_parse_results(
    avg_efficiencies,
    cuts,
    efficiencies,
    fail_cuts,
    fail_efficiencies,
    n_events_with_negative_weights,
    observations_all_events,
    pass_cuts,
    pass_efficiencies,
):
    for n_pass, n_fail, cut in zip(pass_cuts, fail_cuts, cuts):
        logger.info("  %s / %s events pass cut %s", n_pass, n_pass + n_fail, cut)
    for n_pass, n_fail, efficiency in zip(pass_efficiencies, fail_efficiencies, efficiencies):
        logger.info("  %s / %s events pass efficiency %s", n_pass, n_pass + n_fail, efficiency)
    for n_eff, efficiency, n_pass, n_fail in zip(avg_efficiencies, efficiencies, pass_efficiencies, fail_efficiencies):
        logger.info("  average efficiency for %s is %s", efficiency, n_eff / (n_pass + n_fail))

    n_events_pass = len(observations_all_events)

    if len(cuts) > 0:
        logger.info("  %s events pass all cuts/efficiencies", n_events_pass)
    if n_events_with_negative_weights > 0:
        logger.warning("  %s events contain negative weights", n_events_with_negative_weights)

    return n_events_pass


def _parse_event(
    avg_efficiencies,
    cuts: List[Cut],
    efficiencies: List[Efficiency],
    energy_resolutions,
    eta_resolutions,
    fail_cuts,
    fail_efficiencies,
    n_events_with_negative_weights,
    observables: Dict[str, Observable],
    particles,
    pass_cuts,
    pass_efficiencies,
    phi_resolutions,
    pt_resolutions,
    weight_names_all_events,
    weights,
    global_event_data=None,
    print_event=0,
):
    # Negative weights?
    n_events_with_negative_weights = _report_negative_weights(n_events_with_negative_weights, weights)
    if weight_names_all_events is None:
        weight_names_all_events = list(weights.keys())
    weights = np.array(list(weights.values()))

    # Apply smearing
    particles_smeared = _smear_particles(
        particles, energy_resolutions, pt_resolutions, eta_resolutions, phi_resolutions
    )

    # Objects in event
    try:
        variables = _get_objects(
            particles_smeared, particles, pt_resolutions["met"], global_event_data=global_event_data
        )
    except (TypeError, IndexError):
        variables = _get_objects(particles_smeared, particles, met_resolution=None, global_event_data=global_event_data)

    # Observables
    observations, pass_all_observation = _parse_observations(observables, variables)

    # Cuts
    pass_all_cuts = True
    if pass_all_observation:
        pass_all_cuts = _parse_cuts(
            cuts,
            fail_cuts,
            observables,
            observations,
            pass_all_cuts,
            pass_cuts,
            variables,
        )

    # Efficiencies
    pass_all_efficiencies = True
    if pass_all_observation and pass_all_cuts:
        pass_all_efficiencies, total_efficiency = _parse_efficiencies(
            avg_efficiencies,
            efficiencies,
            fail_efficiencies,
            pass_efficiencies,
            variables,
        )

        if pass_all_efficiencies:
            weights *= total_efficiency

    pass_all = pass_all_cuts and pass_all_efficiencies and pass_all_observation

    if print_event > 0:
        logger.debug(
            "Event {} {} observations, {} cuts, {} efficiencies -> {}".format(
                print_event,
                "passes" if pass_all_observation else "FAILS",
                "passes" if pass_all_cuts else "FAILS",
                "passes" if pass_all_efficiencies else "FAILS",
                "passes" if pass_all else "FAILS",
            )
        )
    return n_events_with_negative_weights, observations, pass_all, weight_names_all_events, weights


def _report_negative_weights(n_events_with_negative_weights, weights):
    n_negative_weights = np.sum(np.array(list(weights.values())) < 0.0)

    if n_negative_weights > 0:
        n_events_with_negative_weights += 1
        if n_events_with_negative_weights <= 3:
            logger.warning("Found %s negative weights in event. Weights: %s", n_negative_weights, weights)
        if n_events_with_negative_weights == 3:
            logger.warning("Skipping warnings about negative weights from now on...")

    return n_events_with_negative_weights


def _parse_observations(observables: Dict[str, Observable], variables: Dict[str, list]):
    observations = []
    passed_all = True

    for name, observable in observables.items():
        definition = observable.val_expression
        default = observable.val_default
        required = observable.is_required

        try:
            if isinstance(definition, str):
                value = eval(definition, variables)
            elif isinstance(definition, Callable):
                value = definition(
                    variables["p_truth"],
                    variables["l"],
                    variables["a"],
                    variables["j"],
                    variables["met"],
                )
            else:
                raise TypeError("Not a valid observable")
        except (IndexError, NameError, RuntimeError, SyntaxError, TypeError, ZeroDivisionError):
            passed_all = False if required else True
            value = default if default is not None else np.nan
        finally:
            observations.append(value)

    return observations, passed_all


def _parse_efficiencies(
    avg_efficiencies,
    efficiencies,
    fail_efficiencies,
    pass_efficiencies,
    variables,
):
    # Apply efficiencies
    total_efficiency = 1.0
    pass_all_efficiencies = True

    for i_efficiency, efficiency in enumerate(efficiencies):
        definition = efficiency.val_expression
        default = efficiency.val_default

        try:
            efficiency_result = eval(definition, variables)
            if efficiency_result > 0.0:
                pass_efficiencies[i_efficiency] += 1
                total_efficiency *= efficiency_result
                avg_efficiencies[i_efficiency] += efficiency_result
            else:
                fail_efficiencies[i_efficiency] += 1
                pass_all_efficiencies = False

        except (SyntaxError, NameError, TypeError, ZeroDivisionError, IndexError):
            if default > 0.0:
                pass_efficiencies[i_efficiency] += 1
                total_efficiency *= default
                avg_efficiencies[i_efficiency] += default
            else:
                fail_efficiencies[i_efficiency] += 1
                pass_all_efficiencies = False

    return pass_all_efficiencies, total_efficiency


def _parse_cuts(cuts, fail_cuts, observables, observations, pass_all_cuts, pass_cuts, variables):
    # Objects for cuts
    for obs_name, obs_value in zip(observables.keys(), observations):
        variables[obs_name] = obs_value

    # Check cuts
    for i_cut, cut in enumerate(cuts):
        definition = cut.val_expression
        required = cut.is_required

        try:
            cut_result = eval(definition, variables)
            if cut_result:
                pass_cuts[i_cut] += 1
            else:
                fail_cuts[i_cut] += 1
                pass_all_cuts = False

        except (SyntaxError, NameError, TypeError, ZeroDivisionError, IndexError):
            if required:
                pass_cuts[i_cut] += 1
            else:
                fail_cuts[i_cut] += 1
                pass_all_cuts = False

    return pass_all_cuts


def extract_nuisance_parameters_from_lhe_file(filename: str, systematics: Dict[str, Systematic]):
    """
    Extracts the definition of nuisance parameters from the LHE file
    and returns a systematics_dict with structure:
    {systematics_name: {nuisance_parameter_name: ((benchmark0, weight0), (benchmark1, weight1), processing) }
    """

    logger.debug("Parsing nuisance parameter setup from LHE file at %s", filename)

    # Nuisance parameters (output)
    systematics_dict = OrderedDict()

    # When no systematics setup is defined
    if systematics is None:
        return systematics_dict

    # Parse scale factors from strings in systematics
    logger.debug("Systematics setup: %s", systematics)

    # Unzip and parse LHE file
    initrwgts = _untar_and_parse_lhe_file(filename, ["initrwgt"])

    # Find weight groups
    weight_groups = []
    try:
        for initrwgt in initrwgts:
            weight_groups += initrwgt.findall("weightgroup")
    except KeyError as e:
        raise RuntimeError("Could not find weight groups in LHE file!\n%s", e)
    # if len(weight_groups) == 0:
    #     raise RuntimeError("Zero weight groups in LHE file!")
    logger.debug("%s weight groups", len(weight_groups))

    # Loop over systematics
    for syst_name, syst_obj in systematics.items():
        nuisance_param_dict = _extract_nuisance_param_dict(weight_groups, syst_name, syst_obj)
        systematics_dict[syst_name] = nuisance_param_dict

    return systematics_dict


def _extract_nuisance_param_dict(weight_groups: list, systematics_name: str, systematic: Systematic):
    logger.debug("Extracting nuisance parameter information for systematic %s", systematics_name)

    if systematic.type == SystematicType.NORM:
        nuisance_param_name = f"{systematics_name}_nuisance_param_0"
        benchmark_name = f"{nuisance_param_name}_benchmark_0"
        nuisance_param_definition = (benchmark_name, None), (None, None), systematic.value
        return {nuisance_param_name: nuisance_param_definition}

    elif systematic.type == SystematicType.SCALE:
        # Prepare output
        nuisance_param_definition_parts = []

        # Parse scale variations we need to find
        scale_factors = systematic.value.split(",")
        scale_factors = [float(sf) for sf in scale_factors]

        if len(scale_factors) == 0:
            raise RuntimeError("Cannot parse scale factor string %s", systematic.value)
        elif len(scale_factors) == 1:
            scale_factors = (scale_factors[0],)
        else:
            scale_factors = (scale_factors[-1], scale_factors[0])

        # Loop over scale factors
        for k, scale_factor in enumerate(scale_factors):
            muf = scale_factor if systematic.scale in {SystematicScale.MU, SystematicScale.MUF} else 1.0
            mur = scale_factor if systematic.scale in {SystematicScale.MU, SystematicScale.MUR} else 1.0

            # Loop over weight groups and weights and identify benchmarks
            for wg in weight_groups:
                try:
                    wg_name = wg.attrib["name"]
                except KeyError:
                    logger.warning("New weight group: does not have name attribute, skipping")
                    continue
                logger.debug("New weight group: %s", wg_name)

                if "mg_reweighting" in wg_name.lower() or "scale variation" not in wg_name.lower():
                    continue
                logger.debug("Weight group identified as scale variation")

                weights = wg.findall("weight")

                for weight in weights:
                    try:
                        weight_id = str(weight.attrib["id"])
                        weight_muf = float(weight.attrib["MUF"])
                        weight_mur = float(weight.attrib["MUR"])
                    except KeyError:
                        logger.warning("Scale variation weight does not have all expected attributes")
                        continue

                    logging.debug(
                        "Found scale variation weight %s / muf = %s, mur = %s", weight_id, weight_muf, weight_mur
                    )

                    # Let's skip the entries with a varied dynamical scale for now
                    weight_dynscale = None
                    for key in ["dynscale", "dyn_scale", "DYNSCALE", "DYN_SCALE"]:
                        try:
                            weight_dynscale = int(weight.attrib[key])
                        except KeyError:
                            pass
                    if weight_dynscale is not None:
                        continue

                    # Matching time!
                    if approx_equal(weight_mur, mur) and approx_equal(weight_muf, muf):
                        benchmark_name = f"{systematics_name}_nuisance_param_0_benchmark_{k}"
                        nuisance_param_definition_parts.append((benchmark_name, weight_id))
                        break

        if len(nuisance_param_definition_parts) < len(scale_factors):
            logger.warning(
                "Could not find weights for the scale uncertainty %s in LHE file! The most common source of "
                " this error is not having installed LHAPDF with its Python interface. Please make sure that"
                " you have installed this. You can also check the log file produced by MadGraph for a "
                "warning about this. If LHAPDF is correctly installed and you still get this warning, please"
                " check manually whether the LHE file contains weights from PDF variation, and contact"
                " the MadMiner developer team about this. If you continue with the analysis, MadMiner"
                " will disregard PDF uncertainties.",
                systematics_name,
            )
            return {}
        else:
            # Output
            nuisance_param_name = f"{systematics_name}_nuisance_param_0"
            if len(nuisance_param_definition_parts) > 1:
                nuisance_dict = {
                    nuisance_param_name: (nuisance_param_definition_parts[0], nuisance_param_definition_parts[1], None)
                }
            else:
                nuisance_dict = {nuisance_param_name: (nuisance_param_definition_parts[0], (None, None), None)}
            return nuisance_dict

    elif systematic.type == SystematicType.PDF:
        nuisance_dict = OrderedDict()

        # Loop over weight groups and weights and identify benchmarks
        for wg in weight_groups:
            try:
                wg_name = wg.attrib["name"]
            except KeyError:
                logger.warning("New weight group: does not have name attribute, skipping")
                continue
            logger.debug("New weight group: %s", wg_name)

            if (
                "mg_reweighting" in wg_name.lower()
                or "mwst" not in wg_name.lower()
                or "pdf" not in wg_name.lower()
                or "ct" not in wg_name.lower()
                or systematic.value not in wg_name.lower()
            ):
                continue

            logger.debug("Weight group identified as PDF variation")
            weights = wg.findall("weight")

            for i, weight in enumerate(weights):
                try:
                    weight_id = str(weight.attrib["id"])
                    weight_pdf = int(weight.attrib["PDF"])
                except KeyError:
                    logger.warning("Scale variation weight does not have all expected attributes")
                    continue

                if weight_pdf % 1000 == 0:  # Central element, not eigenvector of covariance matrix
                    logger.debug("Identifying PDF weight %s / %s as central element", weight_id, weight_pdf)
                    continue

                logger.debug("Found PDF weight %s / %s", weight_id, weight_pdf)

                # Add every PDF Hessian direction to nuisance parameters
                nuisance_param_name = f"{systematics_name}_nuisance_param_{i}"
                benchmark_name = f"{nuisance_param_name}_benchmark_0"
                nuisance_dict[nuisance_param_name] = (benchmark_name, weight_id), (None, None), None

        # Check that everything was found
        if len(nuisance_dict) < 0:
            logger.warning(
                "Could not find weights for the PDF uncertainties in LHE file! The most common source of this"
                " error is not having installed LHAPDF with its Python interface. Please make sure that you "
                " have installed this. You can also check the log file produced by MadGraph for a warning"
                " about this. If LHAPDF is correctly installed and you still get this warning, please check"
                " manually whether the LHE file at %s contains weights from PDF variation, and contact"
                " the MadMiner developer team about this. If you continue with the analysis, MadMiner"
                " will disregard PDF uncertainties."
            )
        return nuisance_dict

    else:
        raise RuntimeError("Unknown systematics type %s", systematic.type)


def _parse_xml_event(event, sampling_benchmark):
    # Initialize weights and momenta
    weights = OrderedDict()
    particles = []
    global_event_data = {}

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
    global_event_data["n_particles"] = float(tag_line[0])
    weights[sampling_benchmark] = float(tag_line[2])
    global_event_data["scale"] = float(tag_line[3])
    global_event_data["alpha_qed"] = float(tag_line[4])
    global_event_data["alpha_qcd"] = float(tag_line[5])

    # Parse momenta
    for elements in particle_lines:
        if len(elements) < 10:
            continue
        status = int(elements[1])
        if elements[0] == "#aMCatNLO":
            elements = elements[1:]
        if status == 1:
            pdgid = int(elements[0])
            px = float(elements[6])
            py = float(elements[7])
            pz = float(elements[8])
            e = float(elements[9])
            spin = float(elements[12])
            particle = MadMinerParticle.from_xyzt(px, py, pz, e)
            particle.set_pdgid(pdgid)
            particle.set_spin(spin)
            particles.append(particle)

    # Weights
    if event.find("rwgt") is not None:
        for weight in event.find("rwgt").findall("wgt"):
            weight_id, weight_value = weight.attrib["id"], float(weight.text)
            weights[weight_id] = weight_value

    return particles, weights, global_event_data


def _parse_txt_events(filename, sampling_benchmark):
    # Initialize weights and momenta
    weights = OrderedDict()
    particles = []

    # Some tags so that we know where in the event we are
    do_tag = False
    do_momenta = False
    do_reweight = False

    # Event
    n_events = 0

    reset_event = False

    # Loop through lines in Event
    with open(filename, "r") as file:
        for line in file.readlines():
            # Clean up line
            try:
                line = line.split("#")[0]
            except:
                pass
            line = line.strip()
            elements = line.split()

            # Skip empty/commented out lines
            if len(line) == 0 or len(elements) == 0:
                continue

            # End of LHE file
            elif line == "</LesHouchesEvents>":
                return

            # Beginning of event
            elif line == "<event>":
                # Initialize weights and momenta
                weights = OrderedDict()
                particles = []

                # Some tags so that we know where in the event we are
                do_tag = True
                do_momenta = False
                do_reweight = False

            # End of event
            elif line == "</event>":
                n_events += 1
                yield particles, weights

                # Reset weights and momenta
                weights = OrderedDict()
                particles = []

                # Some tags so that we know where in the event we are
                do_tag = False
                do_momenta = False
                do_reweight = False

            # Beginning of unimportant block
            elif line == "<mgrwt>":
                do_tag = False
                do_momenta = False
                do_reweight = False

            # Beginning of weight block
            elif line == "<rwgt>":
                do_tag = False
                do_momenta = False
                do_reweight = True

            # End of weight block
            elif line == "</rwgt>":
                do_tag = False
                do_momenta = False
                do_reweight = False

            # Read tag -> first weight
            elif do_tag:
                weights[sampling_benchmark] = float(elements[2])

                do_tag = False
                do_momenta = True
                do_reweight = False

            # Read Momenta and store as 4-vector
            elif do_momenta:
                status = int(elements[1])
                if status == 1:
                    pdgid = int(elements[0])
                    px = float(elements[6])
                    py = float(elements[7])
                    pz = float(elements[8])
                    e = float(elements[9])
                    spin = float(elements[12])
                    particle = MadMinerParticle.from_xyzt(px, py, pz, e)
                    particle.set_pdgid(pdgid)
                    particle.set_spin(spin)
                    particles.append(particle)

            # Read reweighted weights
            elif do_reweight:
                rwgtid = line[line.find("<") + 1 : line.find(">")].split("=")[1][1:-1]
                rwgtval = float(line[line.find(">") + 1 : line.find("<", line.find("<") + 1)])
                weights[rwgtid] = rwgtval


def _parse_lhe_file_with_bad_chars(filename):
    # In some cases, the LHE comments can contain bad characters
    with open(filename, "r") as file:
        for line in file:
            comment_pos = line.find("#")
            if comment_pos >= 0:
                yield line[:comment_pos]
            else:
                yield line


def _untar_and_parse_lhe_file(filename, tags=None):

    # Unzip event file
    new_filename = Path(filename).with_suffix("")
    extension = Path(filename).suffix

    if extension == ".gz":
        if not new_filename.exists():
            unzip_file(filename, new_filename)
        filename = new_filename

    for event, elem in ET.iterparse(filename):
        if tags and elem.tag not in tags:
            continue
        else:
            yield elem

        elem.clear()


def _get_objects(particles, particles_truth, met_resolution=None, global_event_data=None):
    # Find visible particles
    electrons = []
    muons = []
    taus = []
    photons = []
    jets = []
    leptons = []
    neutrinos = []
    unstables = []
    invisibles = []

    for particle in particles:
        pdgid = abs(particle.pdgid)
        if pdgid in [1, 2, 3, 4, 5, 6, 9, 21]:
            if pdgid == 5:
                particle.set_tags(False, True, False)
            if pdgid == 6:
                particle.set_tags(False, False, True)
            jets.append(particle)
        elif pdgid == 11:
            electrons.append(particle)
            leptons.append(particle)
        elif pdgid == 13:
            muons.append(particle)
            leptons.append(particle)
        elif pdgid == 15:
            taus.append(particle)
        elif pdgid == 22:
            photons.append(particle)
        elif pdgid in [12, 14, 16]:
            neutrinos.append(particle)
            invisibles.append(particle)
        elif pdgid in [23, 24, 25]:
            unstables.append(particle)
        else:
            logger.warning("Unknown particle with PDG id %s, treating as invisible!")
            invisibles.append(particle)

    # Sort by pT
    electrons = sorted(electrons, reverse=True, key=lambda x: x.pt)
    muons = sorted(muons, reverse=True, key=lambda x: x.pt)
    taus = sorted(taus, reverse=True, key=lambda x: x.pt)
    photons = sorted(photons, reverse=True, key=lambda x: x.pt)
    leptons = sorted(leptons, reverse=True, key=lambda x: x.pt)
    neutrinos = sorted(neutrinos, reverse=True, key=lambda x: x.pt)
    jets = sorted(jets, reverse=True, key=lambda x: x.pt)

    # Sum over all particles
    ht = 0.0
    visible_sum = MadMinerParticle.from_xyzt(0.0, 0.0, 0.0, 0.0)

    standard_ids = set(get_elementary_pdg_ids())
    neutrino_ids = set(
        int(p.pdgid)
        for p in Particle.findall(lambda p: p.pdgid.is_sm_lepton and p.charge == 0)
    )

    for particle in particles:
        if particle.pdgid in standard_ids and particle.pdgid not in neutrino_ids:
            visible_sum += particle
            ht += particle.pt

    # Soft noise
    if met_resolution is not None:
        noise_std = met_resolution[0] + met_resolution[1] * ht
        noise_x = np.random.normal(0.0, noise_std, size=None)
        noise_y = np.random.normal(0.0, noise_std, size=None)
    else:
        noise_x = 0.0
        noise_y = 0.0

    # MET
    met_x = -visible_sum.x + noise_x
    met_y = -visible_sum.y + noise_y
    met = MadMinerParticle.from_xyzt(
        x=met_x,
        y=met_y,
        z=0.0,
        t=(met_x ** 2 + met_y ** 2) ** 0.5,
    )

    # Build objects
    objects = math_commands()
    objects.update(
        {
            "p": particles,
            "p_truth": particles_truth,
            "e": electrons,
            "j": jets,
            "a": photons,
            "mu": muons,
            "tau": taus,
            "l": leptons,
            "met": met,
            "v": neutrinos,
        }
    )

    # Global event_data
    if global_event_data is not None:
        objects.update(global_event_data)

    return objects


def _smear_variable(true_value, resolutions, id):
    """ Adds Gaussian nose to a variable """
    try:
        res = float(resolutions[id][0] + resolutions[id][1] * true_value)

        if res <= 0.0:
            return true_value

        return float(true_value + np.random.normal(0.0, float(res), None))

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

        if (
            pdgid not in energy_resolutions.keys()
            or pdgid not in pt_resolutions.keys()
            or pdgid not in eta_resolutions.keys()
            or pdgid not in phi_resolutions.keys()
        ):
            continue

        if None in energy_resolutions[pdgid] and None in pt_resolutions[pdgid]:
            raise RuntimeError("Cannot derive both pT and energy from on-shell conditions!")

        # Minimum energy and pT
        m = particle.m
        min_e = 0.0
        if None in pt_resolutions[pdgid]:
            min_e = m
        min_pt = 0.0

        # Smear four-momenta
        e = None
        if None not in energy_resolutions[pdgid]:
            e = min_e - 1.0
            while e <= min_e:
                e = _smear_variable(particle.e, energy_resolutions, pdgid)
        pt = None
        if None not in pt_resolutions[pdgid]:
            pt = min_pt - 1.0
            while pt <= min_pt:
                pt = _smear_variable(particle.pt, pt_resolutions, pdgid)
        eta = _smear_variable(particle.eta, eta_resolutions, pdgid)
        phi = _smear_variable(particle.phi, phi_resolutions, pdgid)
        while phi > 2.0 * np.pi:
            phi -= 2.0 * np.pi
        while phi < 0.0:
            phi += 2.0 * np.pi

        if None in energy_resolutions[pdgid]:
            # Calculate E from on-shell conditions
            smeared_particle = MadMinerParticle.from_rhophietatau(pt, phi, eta, particle.m)

        elif None in pt_resolutions[pdgid]:
            # Calculate pT from on-shell conditions
            if e > m:
                pt = (e ** 2 - m ** 2) ** 0.5 / np.cosh(eta)
            else:
                pt = 0.0
            smeared_particle = MadMinerParticle.from_rhophietat(pt, phi, eta, e)

        else:
            # Everything smeared manually
            smeared_particle = MadMinerParticle.from_rhophietat(pt, phi, eta, e)

        # PDG id (also sets charge)
        smeared_particle.set_pdgid(pdgid)
        smeared_particles.append(smeared_particle)

    return smeared_particles


def get_elementary_pdg_ids():
    """Get Standard Model elementary particle IDs"""
    pdg_ids = Particle.findall(
        lambda p: (
            p.pdgid.is_sm_quark
            or p.pdgid.is_sm_lepton
            or p.pdgid.is_sm_gauge_boson_or_higgs
        )
    )

    return [int(p.pdgid) for p in pdg_ids]
