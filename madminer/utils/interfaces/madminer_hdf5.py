from __future__ import absolute_import, division, print_function, unicode_literals

import six
import shutil
import h5py
import numpy as np
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


def save_madminer_settings(
    filename,
    parameters,
    benchmarks,
    benchmarks_is_nuisance=None,
    morphing_components=None,
    morphing_matrix=None,
    systematics=None,
    overwrite_existing_files=True,
):
    """Saves all MadMiner settings into an HDF5 file."""

    parameter_names = _save_parameters(filename, overwrite_existing_files, parameters)
    _save_benchmarks(benchmarks, benchmarks_is_nuisance, filename, parameter_names)
    _save_morphing(filename, morphing_components, morphing_matrix)
    _save_systematics(filename, systematics)


def save_nuisance_setup_to_madminer_file(
    filename,
    weight_names,
    nuisance_parameters,
    reference_benchmark=None,
    sort=True,
    copy_from=None,
    overwrite_existing_nuisance_parameters=True,
):
    """ Saves the names of nuisance-defined benchmarks in an HDF5 file """

    # Copy file
    if copy_from is not None:
        try:
            shutil.copyfile(copy_from, filename)
        except IOError:
            pass

    _save_nuisance_parameters(filename, nuisance_parameters, overwrite_existing_nuisance_parameters)
    benchmark_is_nuisance, benchmark_names, benchmark_values, _ = _load_benchmarks(filename, True, None, False)
    benchmark_names, benchmark_values, benchmark_is_nuisance = _process_new_benchmarks(
        benchmark_is_nuisance, benchmark_names, benchmark_values, sort, weight_names
    )
    _save_benchmarks2(benchmark_is_nuisance, benchmark_names, benchmark_values, filename, reference_benchmark)


def save_preformatted_events_to_madminer_file(
    filename, observations, weights, sampling_benchmarks, copy_setup_from, overwrite_existing_samples=True
):
    _copy_madminer_file(copy_setup_from, filename, overwrite_existing_samples)
    _save_events(
        filename, None, observations, overwrite_existing_samples, sampling_benchmarks, weights, preformatted=True
    )


def save_events_to_madminer_file(
    filename,
    observables,
    observations,
    weights,
    sampling_benchmarks=None,
    n_events_per_sampling_benchmark=None,
    n_events_background=None,
    copy_from=None,
    overwrite_existing_samples=True,
):
    _copy_madminer_file(copy_from, filename, overwrite_existing_samples)

    observable_names = _save_observables(filename, observables, overwrite_existing_samples)

    if weights is None or observations is None:
        return

    _, benchmark_names, _, _ = _load_benchmarks(filename, True, None, return_dict=False)
    logger.debug("Weight names found in event file: %s", [key for key in weights])
    logger.debug("Benchmarks found in MadMiner file: %s", benchmark_names)
    weights_sorted = _sort_weights(benchmark_names, weights)

    if weights is None or observations is None:
        return
    _save_events(
        filename, observable_names, observations, overwrite_existing_samples, sampling_benchmarks, weights_sorted
    )
    _save_n_events(filename, n_events_background, n_events_per_sampling_benchmark, overwrite_existing_samples)


def save_sample_summary_to_madminer_file(
    filename,
    n_events_per_sampling_benchmark=None,
    n_events_background=None,
    copy_from=None,
    overwrite_existing_samples=True,
):
    _copy_madminer_file(copy_from, filename, overwrite_existing_samples)
    _save_n_events(filename, n_events_background, n_events_per_sampling_benchmark, overwrite_existing_samples)


def load_madminer_settings(filename, include_nuisance_benchmarks=False):
    """ Loads MadMiner settings, observables, and weights from a HDF5 file. """

    parameter_names, parameters = _load_parameters(filename)
    benchmark_is_nuisance, benchmarks, reference_benchmark = _load_benchmarks(
        filename, include_nuisance_benchmarks, parameter_names
    )
    morphing_components, morphing_matrix = _load_morphing(filename)
    observables = _load_observables(filename)
    n_background_events, n_samples, n_signal_events_generated_per_benchmark = _load_n_samples(filename)
    systematics = _load_systematics(filename)
    nuisance_parameters = _load_nuisance_params(filename)

    return (
        parameters,
        benchmarks,
        benchmark_is_nuisance,
        morphing_components,
        morphing_matrix,
        observables,
        n_samples,
        systematics,
        reference_benchmark,
        nuisance_parameters,
        n_signal_events_generated_per_benchmark,
        n_background_events,
    )


def load_benchmarks_from_madminer_file(filename, include_nuisance_benchmarks=False):
    with h5py.File(filename, "r") as f:

        # Benchmarks
        try:
            benchmark_names = f["benchmarks/names"][()]
            benchmark_names = [bname.decode("ascii") for bname in benchmark_names]

            logger.debug("Benchmarks found in MadMiner file: %s", benchmark_names)

        except KeyError:
            raise IOError("Cannot read benchmarks from HDF5 file")

        if not include_nuisance_benchmarks:
            try:
                benchmark_is_nuisance = f["benchmarks/is_nuisance"][()]
                benchmark_is_nuisance = [False if is_nuisance == 0 else True for is_nuisance in benchmark_is_nuisance]
            except KeyError:
                logger.info(
                    "HDF5 file does not contain is_nuisance field. Assuming is_nuisance=False for all benchmarks."
                )
                benchmark_is_nuisance = [False for _ in benchmark_names]

            phys_benchmark_names = []
            for name, is_nuisance in zip(benchmark_names, benchmark_is_nuisance):
                if not is_nuisance:
                    phys_benchmark_names.append(name)

            return phys_benchmark_names

    return benchmark_names


def madminer_event_loader(
    filename,
    start=0,
    end=None,
    batch_size=100000,
    include_nuisance_parameters=True,
    benchmark_is_nuisance=None,
    sampling_benchmark=None,
    sampling_factors=None,
    return_sampling_ids=False,
):
    if start is None:
        start = 0

    # Nuisance parameter filtering
    if not include_nuisance_parameters:
        if benchmark_is_nuisance is None:
            logger.warning(
                "include_nuisance_parameters=False without benchmark_is_nuisance information. Returning all weights."
            )
            include_nuisance_parameters = True
        else:
            benchmark_filter = np.logical_not(np.array(benchmark_is_nuisance, dtype=np.bool))

    with h5py.File(filename, "r") as f:

        # Handles to data
        try:
            observations = f["samples/observations"]
            weights = f["samples/weights"]
        except KeyError:
            logger.warning("No events found!")
            return
        try:
            sampling_ids = f["samples/sampling_benchmarks"]
        except KeyError:
            sampling_ids = None
            logger.debug(
                "sampling_benchmarks field not found, probably due to the data being generated by an older"
                " MadMiner version"
            )

        # Preparations
        n_samples = observations.shape[0]
        if weights.shape[0] != n_samples:
            raise ValueError("Number of weights and observations don't match: {}, {}", weights.shape[0], n_samples)

        if end is None:
            end = n_samples
        end = min(n_samples, end)

        if batch_size is None:
            batch_size = n_samples

        current = start

        # Loop over data
        while current < end:
            this_end = min(current + batch_size, end)

            this_observations = np.array(observations[current:this_end])
            if include_nuisance_parameters:
                this_weights = np.array(weights[current:this_end])
            else:
                this_weights = np.array(weights[current:this_end, benchmark_filter])

            if sampling_ids is not None:
                this_sampling_ids = np.array(sampling_ids[current:this_end])

                # Only return data matching sampling_benchmark
                if sampling_benchmark is not None:
                    cut = np.logical_or(this_sampling_ids == sampling_benchmark, this_sampling_ids < 0)

                    this_observations = this_observations[cut]
                    this_weights = this_weights[cut]
                    this_sampling_ids = this_sampling_ids[cut]

                # Rescale weights based on sampling
                if sampling_benchmark is None and sampling_factors is not None:
                    logger.debug("Sampling IDs: %s", this_sampling_ids)
                    k_factors = sampling_factors[this_sampling_ids]
                    logger.debug("k-factors: %s", k_factors)
                    this_weights = k_factors[:, np.newaxis] * this_weights

            else:
                this_sampling_ids = None

            if len(this_observations) > 0:
                if return_sampling_ids:
                    yield (this_observations, this_weights, this_sampling_ids)
                else:
                    yield (this_observations, this_weights)

            current += batch_size


def _save_parameters(filename, overwrite_existing_files, parameters):
    io_tag = "w" if overwrite_existing_files else "x"
    with h5py.File(filename, io_tag) as f:
        # Prepare parameters
        parameter_names = [pname for pname in parameters]
        n_parameters = len(parameter_names)
        parameter_names_ascii = _encode(parameter_names)
        parameter_lha_blocks = _encode([parameters[key][0] for key in parameter_names])
        parameter_lha_ids = np.array([parameters[key][1] for key in parameter_names], dtype=np.int)
        parameter_max_power = np.array([parameters[key][2] for key in parameter_names], dtype=np.int)

        parameter_ranges = np.array([parameters[key][3] for key in parameter_names], dtype=np.float)
        parameter_transforms = []
        for key in parameter_names:
            parameter_transform = parameters[key][4]
            if parameter_transform is None:
                parameter_transform = ""
            parameter_transforms.append(parameter_transform.encode("ascii", "ignore"))

        # Store parameters
        f.create_dataset("parameters/names", (n_parameters,), dtype="S256", data=parameter_names_ascii)
        f.create_dataset("parameters/lha_blocks", (n_parameters,), dtype="S256", data=parameter_lha_blocks)
        f.create_dataset("parameters/lha_ids", data=parameter_lha_ids)
        f.create_dataset("parameters/max_power", data=parameter_max_power)
        f.create_dataset("parameters/ranges", data=parameter_ranges)
        f.create_dataset("parameters/transforms", (n_parameters,), dtype="S256", data=parameter_transforms)
    return parameter_names


def _save_benchmarks(benchmarks, benchmarks_is_nuisance, filename, parameter_names):
    io_tag = "a"  # Read-write if file exists, otherwise create
    with h5py.File(filename, io_tag) as f:
        # Prepare benchmarks
        benchmark_names = [bname for bname in benchmarks]
        n_benchmarks = len(benchmark_names)
        benchmark_names_ascii = _encode(benchmark_names)
        benchmark_values = np.array(
            [[benchmarks[bname][pname] for pname in parameter_names] for bname in benchmark_names]
        )
        if benchmarks_is_nuisance is None:
            benchmark_is_nuisance = [False for _ in benchmarks]
        benchmark_is_nuisance = np.array(
            [1 if is_nuisance else 0 for is_nuisance in benchmark_is_nuisance], dtype=np.int
        )

        # Store benchmarks
        f.create_dataset("benchmarks/names", (n_benchmarks,), dtype="S256", data=benchmark_names_ascii)
        f.create_dataset("benchmarks/values", data=benchmark_values)
        f.create_dataset("benchmarks/is_nuisance", data=benchmark_is_nuisance)


def _save_benchmarks2(benchmark_is_nuisance, benchmark_names, benchmark_values, filename, reference_benchmark):
    io_tag = "a"  # Read-write if file exists, otherwise create
    with h5py.File(filename, io_tag) as f:
        # Prepare benchmarks for saving
        n_benchmarks = len(benchmark_names)
        benchmark_names_ascii = [bname.encode("ascii", "ignore") for bname in benchmark_names]
        benchmark_values = np.array(benchmark_values)
        benchmark_is_nuisance = np.array(
            [1 if is_nuisance else 0 for is_nuisance in benchmark_is_nuisance], dtype=np.int
        )
        benchmark_is_reference = np.array(
            [1 if bname == reference_benchmark else 0 for bname in benchmark_names], dtype=np.int
        )

        logger.debug("Combined benchmark names: %s", benchmark_names)
        logger.debug("Combined is_nuisance: %s", benchmark_is_nuisance)
        logger.debug("Combined is_reference: %s", benchmark_is_reference)

        # Make room for saving all this glorious data
        del f["benchmarks"]

        # Store benchmarks
        f.create_dataset("benchmarks/names", (n_benchmarks,), dtype="S256", data=benchmark_names_ascii)
        f.create_dataset("benchmarks/values", data=benchmark_values)
        f.create_dataset("benchmarks/is_nuisance", data=benchmark_is_nuisance)
        f.create_dataset("benchmarks/is_reference", data=benchmark_is_reference)


def _save_morphing(filename, morphing_components, morphing_matrix):
    io_tag = "a"  # Read-write if file exists, otherwise create
    with h5py.File(filename, io_tag) as f:
        # Store morphing info
        if morphing_components is not None:
            f.create_dataset("morphing/components", data=morphing_components.astype(np.int))
        if morphing_matrix is not None:
            f.create_dataset("morphing/morphing_matrix", data=morphing_matrix.astype(np.float))


def _save_systematics(filename, systematics):
    io_tag = "a"
    with h5py.File(filename, io_tag) as f:
        # Prepare and store systematics setup
        if systematics is not None:
            systematics_names = [key for key in systematics]
            n_systematics = len(systematics_names)
            systematics_names_ascii = _encode(systematics_names)
            systematics_values = _encode([systematics[key] for key in systematics_names])

            f.create_dataset("systematics/names", (n_systematics,), dtype="S256", data=systematics_names_ascii)
            f.create_dataset("systematics/values", (n_systematics,), dtype="S256", data=systematics_values)


def _save_nuisance_parameters(filename, nuisance_parameters, overwrite_existing_nuisance_parameters):
    io_tag = "a"  # Read-write if file exists, otherwise create
    with h5py.File(filename, io_tag) as f:
        # Make space for nuisance params
        if overwrite_existing_nuisance_parameters:
            try:
                del f["nuisance_parameters"]
            except Exception:
                pass

        if nuisance_parameters is not None and len(nuisance_parameters) > 0:
            # Prepare nuisance parameters
            nuisance_names = [pname for pname in nuisance_parameters]
            n_nuisance_params = len(nuisance_names)
            nuisance_names_ascii = _encode(nuisance_names)

            nuisance_benchmarks_pos = [
                "" if nuisance_parameters[key][0] is None else nuisance_parameters[key][0] for key in nuisance_names
            ]
            nuisance_benchmarks_pos = _encode(nuisance_benchmarks_pos)

            nuisance_benchmarks_neg = [
                "" if nuisance_parameters[key][1] is None else nuisance_parameters[key][1] for key in nuisance_names
            ]
            nuisance_benchmarks_neg = _encode(nuisance_benchmarks_neg)

            # Save nuisance parameters
            f.create_dataset("nuisance_parameters/names", (n_nuisance_params,), dtype="S256", data=nuisance_names_ascii)
            f.create_dataset(
                "nuisance_parameters/benchmark_positive",
                (n_nuisance_params,),
                dtype="S256",
                data=nuisance_benchmarks_pos,
            )
            f.create_dataset(
                "nuisance_parameters/benchmark_negative",
                (n_nuisance_params,),
                dtype="S256",
                data=nuisance_benchmarks_neg,
            )


def _save_n_events(filename, n_events_background, n_events_per_sampling_benchmark, overwrite_existing_samples):
    io_tag = "a"  # Read-write if file exists, otherwise create
    with h5py.File(filename, io_tag) as f:
        # Check if groups exist already
        if overwrite_existing_samples:
            try:
                del f["sample_summary"]
            except Exception:
                pass

        if n_events_per_sampling_benchmark is not None:
            f.create_dataset(
                "sample_summary/signal_events_per_benchmark",
                data=np.array(n_events_per_sampling_benchmark, dtype=np.int),
            )
        if n_events_background is not None:
            f.create_dataset("sample_summary/background_events", data=np.array(n_events_background, dtype=np.int))


def _save_events(
    filename,
    observable_names,
    observations,
    overwrite_existing_samples,
    sampling_benchmarks,
    weights,
    preformatted=False,
):
    io_tag = "a"  # Read-write if file exists, otherwise create
    with h5py.File(filename, io_tag) as f:
        # Check if groups exist already
        if overwrite_existing_samples:
            try:
                del f["samples"]
            except Exception:
                pass

        # Save weights
        if not preformatted:
            weights = np.array(weights)
            weights = weights.T  # Shape (n_events, n_weights)
        f.create_dataset("samples/weights", data=weights)

        # Save observable values
        if not preformatted:
            observations = np.array([observations[oname] for oname in observable_names]).T
        f.create_dataset("samples/observations", data=observations)

        if sampling_benchmarks is not None:
            f.create_dataset("samples/sampling_benchmarks", data=np.array(sampling_benchmarks, dtype=np.int))


def _save_observables(filename, observables, overwrite_existing_samples):
    io_tag = "a"  # Read-write if file exists, otherwise create
    if observables is None:
        return None

    with h5py.File(filename, io_tag) as f:
        # Check if groups exist already
        if overwrite_existing_samples:
            try:
                del f["observables"]
            except Exception:
                pass

        # Prepare observable definitions
        observable_names = [oname for oname in observables]
        n_observables = len(observable_names)
        observable_names_ascii = [oname.encode("ascii", "ignore") for oname in observable_names]
        observable_definitions = []
        for key in observable_names:
            definition = observables[key]
            if isinstance(definition, six.string_types):
                observable_definitions.append(definition.encode("ascii", "ignore"))
            else:
                observable_definitions.append("".encode("ascii", "ignore"))

        # Store observable definitions
        f.create_dataset("observables/names", (n_observables,), dtype="S256", data=observable_names_ascii)
        f.create_dataset("observables/definitions", (n_observables,), dtype="S256", data=observable_definitions)

    return observable_names


def _load_parameters(filename):
    with h5py.File(filename, "r") as f:
        # Parameters
        try:
            parameter_names = f["parameters/names"][()]
            parameter_lha_blocks = f["parameters/lha_blocks"][()]
            parameter_lha_ids = f["parameters/lha_ids"][()]
            parameter_ranges = f["parameters/ranges"][()]
            parameter_max_power = f["parameters/max_power"][()]
            parameter_transforms = f["parameters/transforms"][()]

            parameter_names = [pname.decode("ascii") for pname in parameter_names]
            parameter_lha_blocks = [pblock.decode("ascii") for pblock in parameter_lha_blocks]
            parameter_transforms = [ptrf.decode("ascii") for ptrf in parameter_transforms]
            parameter_transforms = [None if ptrf == "" else ptrf for ptrf in parameter_transforms]
            parameter_max_power = np.array(parameter_max_power)
            if len(parameter_max_power.shape) < 2:
                parameter_max_power = parameter_max_power.reshape((-1, 1))

            parameters = OrderedDict()

            for pname, prange, pblock, pid, p_maxpower, ptrf in zip(
                parameter_names,
                parameter_ranges,
                parameter_lha_blocks,
                parameter_lha_ids,
                parameter_max_power,
                parameter_transforms,
            ):
                parameters[pname] = (pblock, int(pid), tuple(p_maxpower), tuple(prange), ptrf)

        except KeyError:
            raise IOError("Cannot read parameters from HDF5 file")
    return parameter_names, parameters


def _load_benchmarks(filename, include_nuisance_benchmarks, parameter_names, return_dict=True):
    with h5py.File(filename, "r") as f:
        # Benchmarks
        try:
            benchmark_names = f["benchmarks/names"][()]
            benchmark_values = f["benchmarks/values"][()]
        except KeyError:
            raise IOError("Cannot read benchmarks from HDF5 file")

        try:
            benchmark_is_nuisance = f["benchmarks/is_nuisance"][()]
            benchmark_is_nuisance = [False if is_nuisance == 0 else True for is_nuisance in benchmark_is_nuisance]
        except KeyError:
            logger.info("HDF5 file does not contain is_nuisance field. Assuming is_nuisance=False for all benchmarks.")
            benchmark_is_nuisance = [False for _ in benchmark_names]

        benchmark_names = [bname.decode("ascii") for bname in benchmark_names]

        reference_benchmark = None
        try:
            benchmark_is_reference = f["benchmarks/is_reference"][()]
            for is_reference, bname in zip(benchmark_is_reference, benchmark_names):
                if is_reference:
                    reference_benchmark = bname
        except KeyError:
            logger.debug("HDF5 file does not contain is_reference field.")

    if return_dict:
        benchmarks = OrderedDict()
        for bname, bvalue_matrix, is_nuisance in zip(benchmark_names, benchmark_values, benchmark_is_nuisance):
            if include_nuisance_benchmarks or (not is_nuisance):
                bvalues = OrderedDict()
                for pname, pvalue in zip(parameter_names, bvalue_matrix):
                    bvalues[pname] = pvalue
                benchmarks[bname] = bvalues

        return benchmark_is_nuisance, benchmarks, reference_benchmark

    return benchmark_is_nuisance, benchmark_names, benchmark_values, reference_benchmark


def _load_n_samples(filename):
    with h5py.File(filename, "r") as f:
        # Number of samples
        try:
            observations = f["samples/observations"]
            weights = f["samples/weights"]

            n_samples = observations.shape[0]

            if weights.shape[0] != n_samples:
                raise ValueError("Number of weights and observations don't match: {}, {}", weights.shape[0], n_samples)

        except KeyError:
            n_samples = 0

        # Sample summary (number if events generated from each benchmark)
        try:
            n_signal_events_generated_per_benchmark = f["sample_summary/signal_events_per_benchmark"][()]
            n_background_events = int(f["sample_summary/background_events"][()])
        except KeyError:
            n_signal_events_generated_per_benchmark = None
            n_background_events = None
    return n_background_events, n_samples, n_signal_events_generated_per_benchmark


def _load_morphing(filename):
    with h5py.File(filename, "r") as f:
        # Morphing
        try:
            morphing_components = np.asarray(f["morphing/components"][()], dtype=np.int)
            morphing_matrix = np.asarray(f["morphing/morphing_matrix"][()])

        except KeyError:
            morphing_components = None
            morphing_matrix = None
    return morphing_components, morphing_matrix


def _load_nuisance_params(filename):
    with h5py.File(filename, "r") as f:
        # Nuisance parameters
        try:
            nuisance_parameter_names = f["nuisance_parameters/names"][()]
            nusiance_parameter_benchmarks_pos = f["nuisance_parameters/benchmark_positive"][()]
            nusiance_parameter_benchmarks_neg = f["nuisance_parameters/benchmark_negative"][()]

            nuisance_parameter_names = _decode(nuisance_parameter_names)
            nusiance_parameter_benchmarks_pos = _decode(nusiance_parameter_benchmarks_pos)
            nusiance_parameter_benchmarks_neg = _decode(nusiance_parameter_benchmarks_neg)
            nusiance_parameter_benchmarks_neg = [
                None if val == "" else val for val in nusiance_parameter_benchmarks_neg
            ]

            nuisance_parameters = OrderedDict(
                zip(nuisance_parameter_names, zip(nusiance_parameter_benchmarks_pos, nusiance_parameter_benchmarks_neg))
            )

        except KeyError:
            nuisance_parameters = None
    return nuisance_parameters


def _load_systematics(filename):
    with h5py.File(filename, "r") as f:
        # Systematics setup
        try:
            systematics_names = f["systematics/names"][()]
            systematics_values = f["systematics/values"][()]

            systematics_names = _decode(systematics_names)
            systematics_values = _decode(systematics_values)

            systematics = OrderedDict(zip(systematics_names, systematics_values))

        except KeyError:
            systematics = None
    return systematics


def _load_observables(filename):
    with h5py.File(filename, "r") as f:
        # Observables
        try:
            observables = OrderedDict()

            observable_names = f["observables/names"][()]
            observable_names = [oname.decode("ascii") for oname in observable_names]
            observable_definitions = f["observables/definitions"][()]
            observable_definitions = [odef.decode("ascii") for odef in observable_definitions]

            for oname, odef in zip(observable_names, observable_definitions):
                observables[oname] = odef
        except KeyError:
            observables = None
    return observables


def _process_new_benchmarks(benchmark_is_nuisance, benchmark_names, benchmark_values, sort, weight_names):
    # Sort weight names
    if sort:
        weight_names = sorted(weight_names)
    # Add weights not found before

    benchmark_values = list(benchmark_values)

    for weight_name in weight_names:
        if weight_name in benchmark_names:
            logger.debug("Benchmark %s already in benchmark_names_phys", weight_name)
            continue

        logger.debug("Adding nuisance benchmark %s", weight_name)

        benchmark_names.append(weight_name)
        benchmark_is_nuisance.append(True)
        benchmark_values.append(np.zeros_like(benchmark_values[0]))

    return benchmark_names, benchmark_values, benchmark_is_nuisance


def _sort_weights(benchmark_names, weights):
    try:
        # Sort weights: First the benchmarks in the right order, then the rest alphabetically
        weights_sorted = []
        for key in benchmark_names:
            weights_sorted.append(weights[key])
        for key in sorted(weights.keys()):
            if key not in benchmark_names:
                weights_sorted.append(key)

        logger.debug("Sorted benchmarks: %s", benchmark_names)
    except Exception as e:
        logger.warning("Issue matching weight names in HepMC file to benchmark names in MadMiner file:\n%s", e)
        weights_sorted = [weights[key] for key in weights]
    return weights_sorted


def _encode(inputs):
    return [key.encode("ascii", "ignore") for key in inputs]


def _decode(inputs):
    return [key.decode("ascii") for key in inputs]


def _copy_madminer_file(copy_setup_from, filename, overwrite_existing_samples):
    if copy_setup_from is not None:
        try:
            shutil.copyfile(copy_setup_from, filename)
        except IOError:
            if not overwrite_existing_samples:
                raise ()
