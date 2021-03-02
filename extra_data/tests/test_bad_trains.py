import os.path as osp
from tempfile import TemporaryDirectory

import h5py
import pytest

from extra_data import H5File
from extra_data.file_access import FileAccess
from . import make_examples

@pytest.fixture(scope='module')
def agipd_file_tid_high():
    with TemporaryDirectory() as td:
        path = osp.join(td, 'CORR-R9999-AGIPD07-S00000.h5')
        make_examples.make_agipd_example_file(path, format_version='0.5')
        with h5py.File(path, 'r+') as f:
            # Initial train IDs are np.arange(10000, 10250)
            f['INDEX/trainId'][10] = 11000
        yield path

@pytest.fixture(scope='module')
def agipd_file_tid_low():
    with TemporaryDirectory() as td:
        path = osp.join(td, 'CORR-R9999-AGIPD07-S00000.h5')
        make_examples.make_agipd_example_file(path, format_version='0.5')
        with h5py.File(path, 'r+') as f:
            # Initial train IDs are np.arange(10000, 10250)
            f['INDEX/trainId'][20] = 9000
        yield path

@pytest.fixture()
def agipd_file_flag0():
    with TemporaryDirectory() as td:
        path = osp.join(td, 'CORR-R9999-AGIPD07-S00000.h5')
        make_examples.make_agipd_file(path, format_version='1.0')
        with h5py.File(path, 'r+') as f:
            f['INDEX/flag'][30] = 0
        yield path

def test_guess_validity(agipd_file_tid_high, agipd_file_tid_low):
    fa = FileAccess(agipd_file_tid_high)
    assert fa.validity_flag.sum() == 249
    assert not fa.validity_flag[10]

    fa = FileAccess(agipd_file_tid_low)
    assert fa.validity_flag.sum() == 249
    assert not fa.validity_flag[20]

def test_validity_flag(agipd_file_flag0):
    fa = FileAccess(agipd_file_flag0)
    assert fa.validity_flag.sum() == 485
    assert not fa.validity_flag[30]

def test_exc_trainid(agipd_file_tid_high, agipd_file_tid_low, agipd_file_flag0):
    f = H5File(agipd_file_tid_high)
    assert len(f.train_ids) == 249
    assert 11000 not in f.train_ids

    f = H5File(agipd_file_tid_high, inc_suspect_trains=True)
    assert len(f.train_ids) == 250
    assert 11000 in f.train_ids

    f = H5File(agipd_file_tid_low)
    assert len(f.train_ids) == 249
    assert 9000 not in f.train_ids

    f = H5File(agipd_file_tid_low, inc_suspect_trains=True)
    assert len(f.train_ids) == 250
    assert 9000 in f.train_ids

    f = H5File(agipd_file_flag0)
    assert len(f.train_ids) == 485
    assert 10030 not in f.train_ids

    f = H5File(agipd_file_flag0, inc_suspect_trains=True)
    assert len(f.train_ids) == 486
    assert 10030 in f.train_ids

# If the tests above pass, the invalid trains in the different sample files
# are being recognised correctly. So for the tests below, we'll mainly test
# each behaviour on just one of the sample files.

def test_keydata_interface(agipd_file_tid_high):
    f = H5File(agipd_file_tid_high)
    kd = f['SPB_DET_AGIPD1M-1/DET/7CH0:xtdf', 'image.data']
    assert len(kd.train_ids) == 249
    assert kd.shape == (249 * 64, 512, 128)

    fi = H5File(agipd_file_tid_high, inc_suspect_trains=True)
    kdi = fi['SPB_DET_AGIPD1M-1/DET/7CH0:xtdf', 'image.data']
    assert len(kdi.train_ids) == 250
    assert kdi.shape == (250 * 64, 512, 128)

def test_array(agipd_file_tid_low):
    f = H5File(agipd_file_tid_low)
    arr = f['SPB_DET_AGIPD1M-1/DET/7CH0:xtdf', 'image.pulseId'].xarray()
    assert arr.shape == (249 * 64, 1)

    fi = H5File(agipd_file_tid_low, inc_suspect_trains=True)
    arri = fi['SPB_DET_AGIPD1M-1/DET/7CH0:xtdf', 'image.pulseId'].xarray()
    assert arri.shape == (250 * 64, 1)

def test_dask_array(agipd_file_flag0):
    f = H5File(agipd_file_flag0)
    arr = f['SPB_DET_AGIPD1M-1/DET/0CH0:xtdf', 'image.pulseId'].dask_array()
    assert arr.shape == (485 * 64, 1)

    fi = H5File(agipd_file_flag0, inc_suspect_trains=True)
    arri = fi['SPB_DET_AGIPD1M-1/DET/0CH0:xtdf', 'image.pulseId'].dask_array()
    assert arri.shape == (486 * 64, 1)
