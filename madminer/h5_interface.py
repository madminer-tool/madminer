import h5py
import numpy as np


def save_madminer_file(filename,
                       parameters,
                       benchmarks,
                       morphing_components=None,
                       morphing_matrix=None,
                       overwrite_existing_files=True):
    """
    Saves all MadMiner settings into an HDF5 file.

    :param filename:
    :param parameters:
    :param benchmarks:
    :param morphing_components:
    :param morphing_matrix:
    :param overwrite_existing_files:
    """

    io_tag = 'w' if overwrite_existing_files else 'x'

    with h5py.File(filename, io_tag) as f:

        # Prepare parameters
        parameter_names = [pname for pname in parameters]
        n_parameters = len(parameter_names)
        parameter_names_ascii = [pname.encode("ascii", "ignore") for pname in parameter_names]
        print(parameter_names_ascii)
        parameter_ranges = np.array(
            [parameters[key][3] for key in parameter_names],
            dtype=np.float
        )
        parameter_lha_blocks = [parameters[key][0].encode("ascii", "ignore") for key in parameter_names]
        parameter_lha_ids = np.array(
            [parameters[key][1] for key in parameter_names],
            dtype=np.int
        )

        # Store parameters
        f.create_dataset('parameters/names', (n_parameters,), dtype='S256', data=parameter_names_ascii)
        f.create_dataset('parameters/ranges', data=parameter_ranges)
        f.create_dataset('parameters/lha_blocks', (n_parameters,), dtype='S256', data=parameter_lha_blocks)
        f.create_dataset("parameters/lha_ids", data=parameter_lha_ids)

        # Prepare benchmarks
        benchmark_names = [bname for bname in benchmarks]
        n_benchmarks = len(benchmark_names)
        benchmark_names_ascii = [bname.encode("ascii", "ignore") for bname in benchmark_names]
        benchmark_values = np.array(
            [
                [benchmarks[bname][pname] for pname in parameter_names]
                for bname in benchmark_names
            ]
        )

        # Store benchmarks
        f.create_dataset('benchmarks/names', (n_benchmarks,), dtype='S256', data=benchmark_names_ascii)
        f.create_dataset('benchmarks/values', data=benchmark_values)

        # Store morphing info
        if morphing_components is not None:
            f.create_dataset("morphing/components", data=morphing_components.astype(np.int))
        if morphing_matrix is not None:
            f.create_dataset("morphing/morphing_matrix", data=morphing_matrix.astype(np.float))
