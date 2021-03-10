import os.path as osp

import h5py
import numpy as np
import pytest
from tempfile import TemporaryDirectory

from . import make_examples


@pytest.fixture(scope='session', params=['0.5', '1.0'])
def format_version(request):
    return request.param


@pytest.fixture(scope='module')
def mock_agipd_data(format_version):
    # This one uses the older index format
    # (first/last/status instead of first/count)
    with TemporaryDirectory() as td:
        path = osp.join(td, 'CORR-R9999-AGIPD07-S00000.h5')
        make_examples.make_agipd_example_file(path, format_version=format_version)
        yield path


@pytest.fixture(scope='module')
def mock_lpd_data(format_version):
    with TemporaryDirectory() as td:
        path = osp.join(td, 'RAW-R9999-LPD00-S00000.h5')
        make_examples.make_lpd_file(path, format_version=format_version)
        yield path


@pytest.fixture(scope='module')
def mock_fxe_control_data(format_version):
    with TemporaryDirectory() as td:
        path = osp.join(td, 'RAW-R0450-DA01-S00001.h5')
        make_examples.make_fxe_da_file(path, format_version=format_version)
        yield path


@pytest.fixture(scope='module')
def mock_sa3_control_data(format_version):
    with TemporaryDirectory() as td:
        path = osp.join(td, 'RAW-R0450-DA01-S00001.h5')
        make_examples.make_sa3_da_file(path, format_version=format_version)
        yield path


@pytest.fixture(scope='module')
def mock_spb_control_data_badname(format_version):
    with TemporaryDirectory() as td:
        path = osp.join(td, 'RAW-R0309-DA01-S00000.h5')
        make_examples.make_data_file_bad_device_name(path, format_version=format_version)
        yield path


@pytest.fixture(scope='session')
def mock_fxe_raw_run(format_version):
    with TemporaryDirectory() as td:
        make_examples.make_fxe_run(td, format_version=format_version)
        yield td


@pytest.fixture(scope='session')
def mock_lpd_parallelgain_run():
    with TemporaryDirectory() as td:
        make_examples.make_lpd_parallelgain_run(td, format_version='1.0')
        yield td


@pytest.fixture(scope='session')
def mock_spb_proc_run(format_version):
    with TemporaryDirectory() as td:
        make_examples.make_spb_run(td, raw=False, format_version=format_version)
        yield td


@pytest.fixture(scope='session')
def mock_reduced_spb_proc_run(format_version):
    """Varying number of frames stored from AGIPD"""
    rng = np.random.RandomState(123)  # Fix seed
    with TemporaryDirectory() as td:
        make_examples.make_reduced_spb_run(td, raw=False, rng=rng,
                                           format_version=format_version)
        yield td


@pytest.fixture(scope='session')
def mock_spb_raw_run(format_version):
    with TemporaryDirectory() as td:
        make_examples.make_spb_run(td, format_version=format_version)
        yield td


@pytest.fixture(scope='session')
def mock_jungfrau_run():
    with TemporaryDirectory() as td:
        make_examples.make_jungfrau_run(td)
        yield td


@pytest.fixture(scope='session')
def mock_scs_run():
    with TemporaryDirectory() as td:
        make_examples.make_scs_run(td)
        yield td


@pytest.fixture(scope='session')
def empty_h5_file():
    with TemporaryDirectory() as td:
        path = osp.join(td, 'empty.h5')
        with h5py.File(path, 'w'):
            pass

        yield path

@pytest.fixture(scope='session')
def mock_empty_file():
    with TemporaryDirectory() as td:
        path = osp.join(td, 'RAW-R0450-DA01-S00002.h5')
        make_examples.make_sa3_da_file(path, ntrains=0)
        yield path
