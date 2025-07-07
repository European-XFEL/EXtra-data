import os
import os.path as osp

import h5py
import numpy as np
import pytest
from tempfile import TemporaryDirectory
import yaml

from . import make_examples


# Make Pyyaml write tuples as regular lists, otherwise it adds a tag to the
# element that makes it impossible to deserialize without a custom constructor.
def tuple_representer(dumper, data):
    return dumper.represent_list(list(data))

class SafeDumperWithTuples(yaml.SafeDumper):
     pass

yaml.add_representer(tuple, tuple_representer, Dumper=SafeDumperWithTuples)


@pytest.fixture(scope='session', params=['0.5', '1.0', '1.2'])
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
def mock_fxe_control_data1(format_version):
    with TemporaryDirectory() as td:
        path = osp.join(td, 'RAW-R0451-DA01-S00001.h5')
        make_examples.make_fxe_da_file(path, firsttrain=20000, format_version=format_version)
        yield path


@pytest.fixture(scope='module')
def mock_sa3_control_data(format_version):
    with TemporaryDirectory() as td:
        path = osp.join(td, 'RAW-R0450-DA01-S00001.h5')
        make_examples.make_sa3_da_file(path, format_version=format_version)
        yield path


@pytest.fixture
def mock_sa3_control_aliases():
    return {
        'sa3-xgm': 'SA3_XTD10_XGM/XGM/DOOCS',
        'hv': ('SA3_XTD10_XGM/XGM/DOOCS', 'pulseEnergy.wavelengthUsed'),
        'beam-x': ('SA3_XTD10_XGM/XGM/DOOCS', 'beamPosition.ixPos'),
        'beam-y': ('SA3_XTD10_XGM/XGM/DOOCS', 'beamPosition.iyPos'),

        'imgfel-frames': ('SA3_XTD10_IMGFEL/CAM/BEAMVIEW:daqOutput', 'data.image.pixels'),
        'imgfel-frames2': ('SA3_XTD10_IMGFEL/CAM/BEAMVIEW2:daqOutput', 'data.image.pixels'),
        'imgfel-screen-pos': ('SA3_XTD10_IMGFEL/MOTOR/SCREEN', 'actualPosition'),
        'imgfel-filter-pos': ('SA3_XTD10_IMGFEL/MOTOR/FILTER', 'actualPosition'),

        'mcp-adc': 'SA3_XTD10_MCP/ADC/1',
        'mcp-mpod': 'SA3_XTD10_MCP/MCPS/MPOD',
        'mcp-voltage': ('SA3_XTD10_MCP/MCPS/MPOD', 'channels.U3.voltage'),
        'mcp-trace': ('SA3_XTD10_MCP/ADC/1:channel_5.output', 'data.rawData'),

        'bogus-source': 'SA4_XTD20_XGM/XGM/DOOCS',
        'bogus-key': ('SA3_XTD10_XGM/XGM/DOOCS', 'foo')
    }

@pytest.fixture
def mock_sa3_control_aliases_yaml(mock_sa3_control_aliases, tmp_path):
    aliases_path = tmp_path / "mock_sa3_control_aliases.yaml"
    with aliases_path.open("w") as f:
        yaml.dump(mock_sa3_control_aliases, f, Dumper=SafeDumperWithTuples)

    return aliases_path


@pytest.fixture(scope='module')
def mock_control_data_with_empty_source(format_version):
    with TemporaryDirectory() as td:
        path = osp.join(td, 'RAW-R0451-DA01-S00001.h5')
        make_examples.make_da_file_with_empty_source(path, format_version=format_version)
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
def mock_lpd_mini_gap_run():
    with TemporaryDirectory() as td:
        make_examples.make_lpd_run_mini_missed_train(td)
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


@pytest.fixture()
def mock_spb_raw_and_proc_run():
    with TemporaryDirectory() as td:
        prop_dir = osp.join(str(td), 'SPB', '201830', 'p002012')

        # Set up raw
        raw_run_dir = osp.join(prop_dir, 'raw', 'r0238')
        os.makedirs(raw_run_dir)
        make_examples.make_spb_run(raw_run_dir)

        # Set up proc
        proc_run_dir = osp.join(prop_dir, 'proc', 'r0238')
        os.makedirs(proc_run_dir)
        make_examples.make_spb_run(proc_run_dir, raw=False)

        yield td, raw_run_dir, proc_run_dir

@pytest.fixture(scope='session')
def mock_spb_raw_run_fmt1():
    with TemporaryDirectory() as td:
        make_examples.make_spb_run(td, format_version="1.2")
        yield td

@pytest.fixture(scope='session')
def mock_modern_spb_proc_run():
    with TemporaryDirectory() as td:
        make_examples.make_modern_spb_proc_run(td)
        yield td

@pytest.fixture()
def mock_spb_raw_and_modern_proc_run():
    with TemporaryDirectory() as td:
        prop_dir = osp.join(str(td), 'SPB', '201830', 'p002012')

        # Set up raw
        raw_run_dir = osp.join(prop_dir, 'raw', 'r0238')
        os.makedirs(raw_run_dir)
        make_examples.make_spb_run(raw_run_dir)

        # Set up proc
        proc_run_dir = osp.join(prop_dir, 'proc', 'r0238')
        os.makedirs(proc_run_dir)
        make_examples.make_modern_spb_proc_run(proc_run_dir)

        yield td, raw_run_dir, proc_run_dir

@pytest.fixture(scope='session')
def mock_jungfrau_run():
    with TemporaryDirectory() as td:
        make_examples.make_jungfrau_run(td)
        yield td

@pytest.fixture(scope='session')
def mock_fxe_jungfrau_run():
    with TemporaryDirectory() as td:
        make_examples.make_fxe_jungfrau_run(td)
        yield td

@pytest.fixture(scope='session')
def mock_scs_run():
    with TemporaryDirectory() as td:
        make_examples.make_scs_run(td)
        yield td


@pytest.fixture(scope='session')
def mock_remi_run():
    with TemporaryDirectory() as td:
        make_examples.make_remi_run(td)
        yield td


@pytest.fixture(scope='session')
def empty_h5_file():
    with TemporaryDirectory() as td:
        path = osp.join(td, 'empty.h5')
        with h5py.File(path, 'w'):
            pass

        yield path


@pytest.fixture(scope='session')
def mock_no_metadata_file():
    with TemporaryDirectory() as td:
        path = osp.join(td, 'no_metadata.h5')
        with h5py.File(path, 'w') as f:
            f.create_dataset('INDEX/trainId', data=[], dtype=np.uint64)

        yield path


@pytest.fixture(scope='session')
def mock_empty_file():
    with TemporaryDirectory() as td:
        path = osp.join(td, 'RAW-R0450-DA01-S00002.h5')
        make_examples.make_sa3_da_file(path, ntrains=0)
        yield path
