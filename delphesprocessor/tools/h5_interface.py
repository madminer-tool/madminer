from __future__ import absolute_import, division, print_function, unicode_literals

import h5py
import numpy as np
import shutil
import logging


def load_benchmarks_from_madminer_file(filename):
    with h5py.File(filename, 'r') as f:

        # Benchmarks
        try:
            benchmark_names = f['benchmarks/names'][()]
            benchmark_names = [bname.decode("ascii") for bname in benchmark_names]

            logging.debug('Benchmarks found in MadMiner file: %s', benchmark_names)

        except KeyError:
            raise IOError('Cannot read benchmarks from HDF5 file')

        return benchmark_names


def add_events_to_madminer_file(filename,
                                observables,
                                observations,
                                weights,
                                copy_from=None,
                                overwrite_existing_samples=True):
    if copy_from is not None:
        try:
            shutil.copyfile(copy_from, filename)
        except IOError:
            if not overwrite_existing_samples:
                raise ()

    io_tag = 'a'  # Read-write if file exists, otherwise create

    with h5py.File(filename, io_tag) as f:

        # Check if groups exist already
        if overwrite_existing_samples:
            try:
                del f['observables']
                del f['samples']
            except:
                pass

        # Prepare observable definitions
        observable_names = [oname for oname in observables]
        n_observables = len(observable_names)
        observable_names_ascii = [oname.encode("ascii", "ignore") for oname in observable_names]
        observable_definitions = [observables[key].encode("ascii", "ignore") for key in observable_names]

        # Store observable definitions
        f.create_dataset('observables/names', (n_observables,), dtype='S256', data=observable_names_ascii)
        f.create_dataset('observables/definitions', (n_observables,), dtype='S256', data=observable_definitions)

        # Try to find benchmarks in file
        logging.debug('Weight names found in Delphes files: %s', [key for key in weights])
        try:
            benchmark_names = f['benchmarks/names'][()]
            benchmark_names = [bname.decode("ascii") for bname in benchmark_names]

            logging.debug('Benchmarks found in MadMiner file: %s', benchmark_names)

            weights_sorted = [weights[key] for key in benchmark_names]

        except Exception as e:
            logging.warning('Issue matching weight names in HepMC file to benchmark names in MadMiner file:\n%s', e)

            weights_sorted = [weights[key] for key in weights]

        # Save weights
        weights_sorted = np.array(weights_sorted)
        weights_sorted = weights_sorted.T  # Shape (n_events, n_benchmarks)
        f.create_dataset("samples/weights", data=weights_sorted)

        # Prepare observable values
        observations = np.array(
            [observations[oname] for oname in observable_names]
        ).T
        f.create_dataset("samples/observations", data=observations)
