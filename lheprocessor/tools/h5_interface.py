from __future__ import absolute_import, division, print_function, unicode_literals

import h5py
import numpy as np
import shutil


def save_madminer_file(filename,
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
                raise()

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

        # Save weights
        weights_event_benchmark = weights.T
        f.create_dataset("samples/weights", data=weights_event_benchmark)

        # Prepare observable values
        observations = np.array(
            [observations[oname] for oname in observable_names]
        ).T
        f.create_dataset("samples/observations", data=observations)

def load_madminer_settings(filename):
    """ Loads MadMiner settings, observables, and weights from a HDF5 file. """
    
    with h5py.File(filename, 'r') as f:
        
        # Parameters
        try:
            parameter_names = f['parameters/names'][()]
            parameter_lha_blocks = f['parameters/lha_blocks'][()]
            parameter_lha_ids = f['parameters/lha_ids'][()]
            parameter_ranges = f['parameters/ranges'][()]
            parameter_max_power = f['parameters/max_power'][()]
            
            parameter_names = [pname.decode("ascii") for pname in parameter_names]
            parameter_lha_blocks = [pblock.decode("ascii") for pblock in parameter_lha_blocks]
            
            parameters = OrderedDict()
            
            for pname, prange, pblock, pid, p_maxpower in zip(parameter_names, parameter_ranges, parameter_lha_blocks,
                                                              parameter_lha_ids, parameter_max_power):
                parameters[pname] = (
                                     pblock,
                                     int(pid),
                                     int(p_maxpower),
                                     tuple(prange)
                                     )
    
        except KeyError:
            raise IOError('Cannot read parameters from HDF5 file')

def load_benchmark_names(filename):
    """ Loads MadMiner Benchmarks Names from a HDF5 file. """
    
    with h5py.File(filename, 'r') as f:

        # Benchmarks
        try:
            benchmark_names = f['benchmarks/names'][()]
            benchmark_names = [bname.decode("ascii") for bname in benchmark_names]
        except KeyError:
            raise IOError('Cannot read benchmarks from HDF5 file')

        return benchmark_names
