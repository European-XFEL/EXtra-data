import os.path as osp
from pathlib import Path

from h5py import File
import numpy as np
from pytest import fixture, raises
from tempfile import TemporaryDirectory

from extra_data.validation import FileAccess, FileValidator, RunValidator, ValidationError, main
from . import make_examples


@fixture(scope='function')
def agipd_file():
    with TemporaryDirectory() as td:
        path = osp.join(td, 'RAW-R0239-AGIPD00-S00000.h5')
        make_examples.make_agipd_file(path)

        yield path


@fixture(scope='function')
def data_aggregator_file():
    with TemporaryDirectory() as td:
        path = osp.join(td, 'RAW-R0450-DA01-S00001.h5')
        make_examples.make_fxe_da_file(path)

        yield path


def test_validate_run(mock_fxe_raw_run):
    rv = RunValidator(mock_fxe_raw_run)
    rv.validate()


def test_file_error(mock_fxe_raw_run):
    not_readable = Path(mock_fxe_raw_run) / 'notReadable.h5'
    not_readable.touch(mode=0o066)

    problems = RunValidator(mock_fxe_raw_run).run_checks()
    assert len(problems) == 1
    assert problems[0]['msg'] == 'Could not access file'
    assert problems[0]['file'] == str(not_readable)


def test_zeros_in_train_ids(agipd_file):
    with File(agipd_file, 'r+') as f:
        # introduce zeros in trainId
        f['/INDEX/trainId'][12] = 0

    with raises(ValidationError) as excinfo:
        FileValidator(FileAccess(agipd_file)).validate()
    problem = excinfo.value.problems[0]
    assert problem['msg'] == 'Zeroes in trainId index before last train ID'
    assert problem['dataset'] == 'INDEX/trainId'
    assert 'RAW-R0239-AGIPD00-S00000.h5' in problem['file']


def test_non_strictly_increasing_train_ids(agipd_file):
    with File(agipd_file, 'r+') as f:
        # introduce non strictly increasing trainId
        f['/INDEX/trainId'][10] = 11010
        f['/INDEX/trainId'][20] = 5

    with raises(ValidationError) as excinfo:
        FileValidator(FileAccess(agipd_file)).validate()
    problem = excinfo.value.problems.pop()
    assert problem['msg'] == 'Train IDs are not strictly increasing, e.g. at 10 (11010 >= 10011)'
    assert problem['dataset'] == 'INDEX/trainId'
    assert 'RAW-R0239-AGIPD00-S00000.h5' in problem['file']


def test_index_pointing_outside_dataset(data_aggregator_file):
    with File(data_aggregator_file, 'r+') as f:
        # index pointing outside dataset
        f['/INDEX/FXE_XAD_GEC/CAM/CAMERA:daqOutput/data/first'][30] = 999

    with raises(ValidationError) as excinfo:
        FileValidator(FileAccess(data_aggregator_file)).validate()
    assert 'Index referring to data (1000) outside dataset (400)' in str(excinfo.value)


def test_invalid_first_dataset(data_aggregator_file):
    with File(data_aggregator_file, 'a') as f:
        # invalid first shape
        length = len(f['INDEX/SA1_XTD2_XGM/DOOCS/MAIN:output/data/first'])
        f['INDEX/SA1_XTD2_XGM/DOOCS/MAIN:output/data/first'].resize((length+1,))

    with raises(ValidationError) as excinfo:
        FileValidator(FileAccess(data_aggregator_file)).validate()
    problem = excinfo.value.problems.pop()
    assert problem['msg'] == 'Index first & count have different number of entries'
    assert problem['dataset'] == 'INDEX/SA1_XTD2_XGM/DOOCS/MAIN:output/data'
    assert problem['first_shape'] == (401,)
    assert problem['count_shape'] == (400,)
    assert 'RAW-R0450-DA01-S00001.h5' in problem['file']


def test_invalid_first_and_count_dataset(data_aggregator_file):
    with File(data_aggregator_file, 'a') as f:
        # invalid first/index shape
        length = len(f['INDEX/SA1_XTD2_XGM/DOOCS/MAIN:output/data/first'])
        f['INDEX/SA1_XTD2_XGM/DOOCS/MAIN:output/data/first'].resize((length-1,))
        length = len(f['INDEX/SA1_XTD2_XGM/DOOCS/MAIN:output/data/count'])
        f['INDEX/SA1_XTD2_XGM/DOOCS/MAIN:output/data/count'].resize((length-1,))

    with raises(ValidationError) as excinfo:
        FileValidator(FileAccess(data_aggregator_file)).validate()
    problem = excinfo.value.problems.pop()
    assert problem['msg'] == 'Index has wrong number of entries'
    assert problem['dataset'] == 'INDEX/SA1_XTD2_XGM/DOOCS/MAIN:output/data'
    assert problem['index_shape'] == (399,)
    assert problem['trainids_shape'] == (400,)
    assert 'RAW-R0450-DA01-S00001.h5' in problem['file']


def test_first_dataset_not_starting_from_zero(data_aggregator_file):
    with File(data_aggregator_file, 'a') as f:
        # first index not starting at zero
        f['INDEX/SA1_XTD2_XGM/DOOCS/MAIN:output/data/first'][0] = 1

    with raises(ValidationError) as excinfo:
        FileValidator(FileAccess(data_aggregator_file)).validate()
    assert "Index doesn't start at 0" in str(excinfo.value)
    assert "INDEX/SA1_XTD2_XGM/DOOCS/MAIN:output/data" in str(excinfo.value)


def test_overlap(agipd_file):
    with File(agipd_file, 'r+') as f:
        # overlap first index
        f['INDEX/SPB_DET_AGIPD1M-1/DET/0CH0:xtdf/image/first'][1] = 0
        f['INDEX/SPB_DET_AGIPD1M-1/DET/0CH0:xtdf/image/count'][1] = 128  # no gaps

    with raises(ValidationError) as excinfo:
        FileValidator(FileAccess(agipd_file)).validate()
    problem = excinfo.value.problems.pop()
    assert problem['msg'] == 'Overlaps (1) in index, e.g. at 0 (0 + 64 > 0)'
    assert problem['dataset'] == 'INDEX/SPB_DET_AGIPD1M-1/DET/0CH0:xtdf/image'
    assert 'RAW-R0239-AGIPD00-S00000.h5' in problem['file']


def test_gaps(agipd_file):
    with File(agipd_file, 'r+') as f:
        # gap in index
        f['INDEX/SPB_DET_AGIPD1M-1/DET/0CH0:xtdf/image/first'][1] = 0
        f['INDEX/SPB_DET_AGIPD1M-1/DET/0CH0:xtdf/image/count'][0] = 0

    with raises(ValidationError) as excinfo:
        FileValidator(FileAccess(agipd_file)).validate()
    problem = excinfo.value.problems.pop()
    assert problem['msg'] == 'Gaps (1) in index, e.g. at 1 (0 + 64 < 128)'
    assert problem['dataset'] == 'INDEX/SPB_DET_AGIPD1M-1/DET/0CH0:xtdf/image'
    assert 'RAW-R0239-AGIPD00-S00000.h5' in problem['file']


def test_file_without_data(mock_empty_file):
    FileValidator(FileAccess(mock_empty_file)).validate()


def test_control_data_timestamps(data_aggregator_file):
    with File(data_aggregator_file, 'r+') as f:
        # control data timestamp is not in data
        ts = f['CONTROL/SA1_XTD2_XGM/DOOCS/MAIN/pulseEnergy/photonFlux/timestamp']
        ts[:] = np.arange(len(ts)) + 1
        ts[10] = 5

    with raises(ValidationError) as excinfo:
        FileValidator(FileAccess(data_aggregator_file)).validate()
    problem = excinfo.value.problems.pop()
    assert problem['msg'] == 'Timestamp is decreasing, e.g. at 10 (5 < 10)'
    assert problem['dataset'] == 'CONTROL/SA1_XTD2_XGM/DOOCS/MAIN/pulseEnergy/photonFlux/timestamp'
    assert 'RAW-R0450-DA01-S00001.h5' in problem['file']


def test_main_file_non_h5(tmp_path, capsys):
    not_h5 = tmp_path / 'notHDF5.h5'
    not_h5.write_text("Accessible file, not HDF5")

    status = main([str(not_h5)])
    assert status == 1
    assert 'Could not open HDF5 file' in capsys.readouterr().out
