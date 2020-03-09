import os.path as osp

import h5py
import pytest
from tempfile import TemporaryDirectory

from . import make_examples


@pytest.fixture(scope='module', params=['0.5', '1.0'])
def mock_agipd_data(request):
    # This one uses the older index format
    # (first/last/status instead of first/count)
    with TemporaryDirectory() as td:
        path = osp.join(td, 'CORR-R9999-AGIPD07-S00000.h5')
        make_examples.make_agipd_example_file(path, format_version=request.param)
        yield path


@pytest.fixture(scope='module', params=['0.5', '1.0'])
def mock_lpd_data(request):
    with TemporaryDirectory() as td:
        path = osp.join(td, 'RAW-R9999-LPD00-S00000.h5')
        make_examples.make_lpd_file(path, format_version=request.param)
        yield path


@pytest.fixture(scope='module', params=['0.5', '1.0'])
def mock_fxe_control_data(request):
    with TemporaryDirectory() as td:
        path = osp.join(td, 'RAW-R0450-DA01-S00001.h5')
        make_examples.make_fxe_da_file(path, format_version=request.param)
        yield path


@pytest.fixture(scope='module', params=['0.5', '1.0'])
def mock_sa3_control_data(request):
    with TemporaryDirectory() as td:
        path = osp.join(td, 'RAW-R0450-DA01-S00001.h5')
        make_examples.make_sa3_da_file(path, format_version=request.param)
        yield path


@pytest.fixture(scope='module', params=['0.5', '1.0'])
def mock_spb_control_data_badname(request):
    with TemporaryDirectory() as td:
        path = osp.join(td, 'RAW-R0309-DA01-S00000.h5')
        make_examples.make_data_file_bad_device_name(path, format_version=request.param)
        yield path


@pytest.fixture(scope='session', params=['0.5', '1.0'])
def mock_fxe_raw_run(request):
    with TemporaryDirectory() as td:
        make_examples.make_fxe_run(td, format_version=request.param)
        yield td


@pytest.fixture(scope='session', params=['0.5', '1.0'])
def mock_spb_proc_run(request):
    with TemporaryDirectory() as td:
        make_examples.make_spb_run(td, raw=False, format_version=request.param)
        yield td


@pytest.fixture(scope='session', params=['0.5', '1.0'])
def mock_spb_raw_run(request):
    with TemporaryDirectory() as td:
        make_examples.make_spb_run(td, format_version=request.param)
        yield td


@pytest.fixture(scope='session')
def empty_h5_file():
    with TemporaryDirectory() as td:
        path = osp.join(td, 'empty.h5')
        with h5py.File(path, 'w'):
            pass

        yield path
