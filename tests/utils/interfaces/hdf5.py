from pathlib import Path
from tempfile import mkstemp

import pytest

from madminer.models import Observable
from madminer.utils.interfaces.hdf5 import (
    EMPTY_EXPR,
    _load_observables,
    _save_observables,
)


@pytest.fixture(scope="function")
def dummy_hdf5_file() -> str:

    _, file_path = mkstemp(suffix=".h5")

    try:
        yield file_path
    finally:
        Path(file_path).unlink()


def test_saving_observables_with_callable_expression(dummy_hdf5_file: str):
    """
    Tests that callable observables can be (de)serialized into and from a HDF5 file

    Parameters
    ----------
    dummy_hdf5_file: str
        Path to the temporal file to use during the test
    """

    dummy_callable = str
    to_save_names = ["obs_name_1", "obs_name_2"]
    to_save_defs = [dummy_callable, dummy_callable]

    _save_observables(
        file_name=dummy_hdf5_file,
        file_override=True,
        observable_names=to_save_names,
        observable_defs=to_save_defs,
    )

    (
        loaded_names,
        loaded_defs,
    ) = _load_observables(file_name=dummy_hdf5_file)

    assert all(
        o.val_expression == EMPTY_EXPR
        for o in (Observable(n, d) for n, d in zip(loaded_names, loaded_defs))
    )
