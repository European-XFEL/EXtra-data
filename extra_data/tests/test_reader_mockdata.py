from datetime import datetime, timedelta, timezone
from itertools import islice
from multiprocessing import Process
from pathlib import Path
from textwrap import dedent
from warnings import catch_warnings

import h5py
import numpy as np
import os
import pandas as pd
import pytest
import stat
import shutil
from tempfile import mkdtemp
from testpath import assert_isfile
from unittest import mock
from xarray import DataArray

from extra_data import (
    H5File, RunDirectory, by_index, by_id,
    SourceNameError, PropertyNameError, DataCollection, open_run,
    MultiRunError
)
from extra_data.reader import DEFAULT_ALIASES_FILE


def test_iterate_trains(mock_agipd_data, mock_control_data_with_empty_source):
    with H5File(mock_agipd_data) as f:
        for train_id, data in islice(f.trains(), 10):
            assert train_id in range(10000, 10250)
            assert 'SPB_DET_AGIPD1M-1/DET/7CH0:xtdf' in data
            assert len(data) == 1
            assert 'image.data' in data['SPB_DET_AGIPD1M-1/DET/7CH0:xtdf']

    with H5File(mock_control_data_with_empty_source) as f:
        # smoke test
        tid, data = next(f.trains())
        assert list(data['SA3_XTD10_VAC/GAUGE/G30520C'].keys()) == ['metadata']


def test_iterate_trains_flat_keys(mock_agipd_data):
    with H5File(mock_agipd_data) as f:
        for train_id, data in islice(f.trains(flat_keys=True), 10):
            assert train_id in range(10000, 10250)
            assert ('SPB_DET_AGIPD1M-1/DET/7CH0:xtdf', 'image.data') in data


def test_iterate_trains_keep_dims(mock_jungfrau_run):
    run = RunDirectory(mock_jungfrau_run)
    for _, data in islice(run.select(
        '*JF4M/DET/*', 'data.adc'
    ).trains(keep_dims=True), 10):

        assert data[
            'SPB_IRDA_JF4M/DET/JNGFR01:daqOutput']['data.adc'].shape == (
                1, 16, 512, 1024)


def test_get_train_keep_dims(mock_jungfrau_run):
    run = RunDirectory(mock_jungfrau_run)
    _, data = run.select(
        '*JF4M/DET/*', 'data.adc').train_from_index(0, keep_dims=True)
    assert data[
        'SPB_IRDA_JF4M/DET/JNGFR01:daqOutput']["data.adc"].shape == (
            1, 16, 512, 1024)


def test_get_train_bad_device_name(mock_spb_control_data_badname):
    # Check that we can handle devices which don't have the standard Karabo
    # name structure A/B/C.
    with H5File(mock_spb_control_data_badname) as f:
        train_id, data = f.train_from_id(10004)
        assert train_id == 10004
        device = 'SPB_IRU_SIDEMIC_CAM:daqOutput'
        assert device in data
        assert 'data.image.dims' in data[device]
        dims = data[device]['data.image.dims']
        assert list(dims) == [1000, 1000]


def test_detector_info_oldfmt(mock_agipd_data):
    with H5File(mock_agipd_data) as f:
        di = f.detector_info('SPB_DET_AGIPD1M-1/DET/7CH0:xtdf')
        assert di['dims'] == (512, 128)
        assert di['frames_per_train'] == 64
        assert di['total_frames'] == 16000


def test_detector_info(mock_lpd_data):
    with H5File(mock_lpd_data) as f:
        di = f.detector_info('FXE_DET_LPD1M-1/DET/0CH0:xtdf')
        assert di['dims'] == (256, 256)
        assert di['frames_per_train'] == 128
        assert di['total_frames'] == 128 * 480


def test_train_info(mock_lpd_data, capsys):
    with H5File(mock_lpd_data) as f:
        f.train_info(10004)
        out, err = capsys.readouterr()
        assert "Devices" in out
        assert "FXE_DET_LPD1M-1/DET/0CH0:xtdf" in out


def test_info(mock_spb_raw_run):
    run = RunDirectory(mock_spb_raw_run)
    run.info(details_for_sources='*/DOOCS/*')  # Smoketest


def test_iterate_trains_fxe(mock_fxe_control_data):
    with H5File(mock_fxe_control_data) as f:
        for train_id, data in islice(f.trains(), 10):
            assert train_id in range(10000, 10400)
            assert 'SA1_XTD2_XGM/DOOCS/MAIN' in data.keys()
            assert 'beamPosition.ixPos.value' in data['SA1_XTD2_XGM/DOOCS/MAIN']
            assert 'data.image.pixels' in data['FXE_XAD_GEC/CAM/CAMERA:daqOutput']
            assert 'data.image.pixels' not in data['FXE_XAD_GEC/CAM/CAMERA_NODATA:daqOutput']


def test_iterate_file_select_trains(mock_fxe_control_data):
    with H5File(mock_fxe_control_data) as f:
        tids = [tid for (tid, _) in f.trains(train_range=by_id[:10003])]
        assert tids == [10000, 10001, 10002]

        tids = [tid for (tid, _) in f.trains(train_range=by_index[-2:])]
        assert tids == [10398, 10399]


def test_iterate_trains_select_keys(mock_fxe_control_data):
    sel = {
        'SA1_XTD2_XGM/DOOCS/MAIN': {
            'beamPosition.ixPos.value',
            'beamPosition.ixPos.timestamp',
        }
    }

    with H5File(mock_fxe_control_data) as f:
        for train_id, data in islice(f.trains(devices=sel), 10):
            assert train_id in range(10000, 10400)
            assert 'SA1_XTD2_XGM/DOOCS/MAIN' in data.keys()
            assert 'beamPosition.ixPos.value' in data['SA1_XTD2_XGM/DOOCS/MAIN']
            assert 'beamPosition.ixPos.timestamp' in data['SA1_XTD2_XGM/DOOCS/MAIN']
            assert 'beamPosition.iyPos.value' not in data['SA1_XTD2_XGM/DOOCS/MAIN']
            assert 'SA3_XTD10_VAC/TSENS/S30160K' not in data


def test_iterate_trains_require_all(mock_sa3_control_data):
    with H5File(mock_sa3_control_data) as f:
        trains_iter = f.trains(
            devices=[('*/CAM/BEAMVIEW:daqOutput', 'data.image.dims')], require_all=True
        )
        tids = [t for (t, _) in trains_iter]
        assert tids == []
        trains_iter = f.trains(
            devices=[('*/CAM/BEAMVIEW:daqOutput', 'data.image.dims')], require_all=False
        )
        tids = [t for (t, _) in trains_iter]
        assert tids != []


def test_read_fxe_raw_run(mock_fxe_raw_run):
    run = RunDirectory(mock_fxe_raw_run)
    assert len(run.files) == 18  # 16 LPD 1M + 2 control data files
    assert run.train_ids == list(range(10000, 10480))
    run.info()  # Smoke test


def test_read_fxe_raw_run_selective(mock_fxe_raw_run):
    run = RunDirectory(mock_fxe_raw_run, include='*DA*')
    assert run.train_ids == list(range(10000, 10480))
    assert 'SA1_XTD2_XGM/DOOCS/MAIN' in run.control_sources
    assert 'FXE_DET_LPD1M-1/DET/0CH0:xtdf' not in run.detector_sources
    run = RunDirectory(mock_fxe_raw_run, include='*LPD*')
    assert run.train_ids == list(range(10000, 10480))
    assert 'SA1_XTD2_XGM/DOOCS/MAIN' not in run.control_sources
    assert 'FXE_DET_LPD1M-1/DET/0CH0:xtdf' in run.detector_sources


def test_read_spb_proc_run(mock_spb_proc_run):
    run = RunDirectory(mock_spb_proc_run) #Test for calib data
    assert len(run.files) == 16 # only 16 detector modules for calib data
    assert run.train_ids == list(range(10000, 10064)) #64 trains
    tid, data = next(run.trains())
    device = 'SPB_DET_AGIPD1M-1/DET/15CH0:xtdf'
    assert tid == 10000
    for prop in ('image.gain', 'image.mask', 'image.data'):
        assert prop in data[device]
    assert 'u1' == data[device]['image.gain'].dtype
    assert 'u4' == data[device]['image.mask'].dtype
    assert 'f4' == data[device]['image.data'].dtype
    run.info() # Smoke test
    run.plot_missing_data() # Smoke test


def test_iterate_spb_raw_run(mock_spb_raw_run):
    run = RunDirectory(mock_spb_raw_run)
    trains_iter = run.trains()
    tid, data = next(trains_iter)
    assert tid == 10000
    device = 'SPB_IRU_CAM/CAM/SIDEMIC:daqOutput'
    assert device in data
    assert data[device]['data.image.pixels'].shape == (1024, 768)


def test_iterate_spb_raw_run_keep_dims(mock_spb_raw_run):
    run = RunDirectory(mock_spb_raw_run)
    trains_iter = run.select(
        'SPB_IRU_CAM/CAM/SIDEMIC:daqOutput',
        'data.image.pixels').trains(keep_dims=True)
    _, data = next(trains_iter)

    assert data[
        'SPB_IRU_CAM/CAM/SIDEMIC:daqOutput']['data.image.pixels'
    ].shape == (1, 1024, 768)


def test_properties_fxe_raw_run(mock_fxe_raw_run):
    run = RunDirectory(mock_fxe_raw_run)

    assert run.train_ids == list(range(10000, 10480))
    assert 'SPB_XTD9_XGM/DOOCS/MAIN' in run.control_sources
    assert 'FXE_DET_LPD1M-1/DET/15CH0:xtdf' in run.instrument_sources


def test_iterate_fxe_run(mock_fxe_raw_run):
    run = RunDirectory(mock_fxe_raw_run)
    trains_iter = run.trains()
    tid, data = next(trains_iter)
    assert tid == 10000
    assert 'FXE_DET_LPD1M-1/DET/15CH0:xtdf' in data
    assert 'image.data' in data['FXE_DET_LPD1M-1/DET/15CH0:xtdf']
    assert 'FXE_XAD_GEC/CAM/CAMERA' in data
    assert 'firmwareVersion.value' in data['FXE_XAD_GEC/CAM/CAMERA']


def test_iterate_select_trains(mock_fxe_raw_run):
    run = RunDirectory(mock_fxe_raw_run)

    tids = [tid for (tid, _) in run.trains(train_range=by_id[10004:10006])]
    assert tids == [10004, 10005]

    tids = [tid for (tid, _) in run.trains(train_range=by_id[:10003])]
    assert tids == [10000, 10001, 10002]

    # Overlap with start of run
    tids = [tid for (tid, _) in run.trains(train_range=by_id[9000:10003])]
    assert tids == [10000, 10001, 10002]

    # Overlap with end of run
    tids = [tid for (tid, _) in run.trains(train_range=by_id[10478:10500])]
    assert tids == [10478, 10479]

    # Not overlapping
    with catch_warnings(record=True) as w:
        tids = [tid for (tid, _) in run.trains(train_range=by_id[9000:9050])]
        assert tids == []
    assert 'before' in str(w[0].message)

    with catch_warnings(record=True) as w:
        tids = [tid for (tid, _) in run.trains(train_range=by_id[10500:10550])]
        assert tids == []
    assert 'after' in str(w[0].message)

    tids = [tid for (tid, _) in run.trains(train_range=by_index[4:6])]
    assert tids == [10004, 10005]


def test_iterate_run_glob_devices(mock_fxe_raw_run):
    run = RunDirectory(mock_fxe_raw_run)
    trains_iter = run.trains([("*/DET/*", "image.data")])
    tid, data = next(trains_iter)
    assert tid == 10000
    assert 'FXE_DET_LPD1M-1/DET/15CH0:xtdf' in data
    assert 'image.data' in data['FXE_DET_LPD1M-1/DET/15CH0:xtdf']
    assert 'detector.data' not in data['FXE_DET_LPD1M-1/DET/15CH0:xtdf']
    assert 'FXE_XAD_GEC/CAM/CAMERA' not in data


def test_train_by_id(mock_fxe_raw_run, mock_control_data_with_empty_source):
    # full run
    run = RunDirectory(mock_fxe_raw_run)
    _, data = run.train_from_id(10024)
    assert 'FXE_DET_LPD1M-1/DET/15CH0:xtdf' in data
    assert 'image.data' in data['FXE_DET_LPD1M-1/DET/15CH0:xtdf']
    assert 'FXE_XAD_GEC/CAM/CAMERA' in data
    assert 'firmwareVersion.value' in data['FXE_XAD_GEC/CAM/CAMERA']

    # selection
    run = RunDirectory(mock_fxe_raw_run)
    _, data = run.train_from_id(10024, [('*/DET/*', 'image.data')])
    assert 'FXE_DET_LPD1M-1/DET/15CH0:xtdf' in data
    assert 'image.data' in data['FXE_DET_LPD1M-1/DET/15CH0:xtdf']
    assert 'FXE_XAD_GEC/CAM/CAMERA' not in data

    # missing control data
    with H5File(mock_control_data_with_empty_source) as f:
        _, data = f.train_from_id(10000)
        assert 'SA3_XTD10_VAC/GAUGE/G30520C' in data
        assert ['metadata'] == list(data['SA3_XTD10_VAC/GAUGE/G30520C'].keys())


def test_train_from_index_fxe_run(mock_fxe_raw_run):
    run = RunDirectory(mock_fxe_raw_run)
    _, data = run.train_from_index(479)
    assert 'FXE_DET_LPD1M-1/DET/15CH0:xtdf' in data
    assert 'image.data' in data['FXE_DET_LPD1M-1/DET/15CH0:xtdf']
    assert 'FXE_XAD_GEC/CAM/CAMERA' in data
    assert 'firmwareVersion.value' in data['FXE_XAD_GEC/CAM/CAMERA']


def test_file_get_series_control(mock_fxe_control_data):
    with H5File(mock_fxe_control_data) as f:
        s = f.get_series('SA1_XTD2_XGM/DOOCS/MAIN', "beamPosition.iyPos.value")
        assert isinstance(s, pd.Series)
        assert len(s) == 400
        assert s.index[0] == 10000


def test_file_get_series_instrument(mock_spb_proc_run):
    agipd_file = os.path.join(mock_spb_proc_run, 'CORR-R0238-AGIPD07-S00000.h5')
    with H5File(agipd_file) as f:
        s = f.get_series('SPB_DET_AGIPD1M-1/DET/7CH0:xtdf', 'header.linkId')
        assert isinstance(s, pd.Series)
        assert len(s) == 64
        assert s.index[0] == 10000

        # Multiple readings per train
        s2 = f.get_series('SPB_DET_AGIPD1M-1/DET/7CH0:xtdf', 'image.pulseId')
        assert isinstance(s2, pd.Series)
        assert not s2.index.is_unique
        assert len(s2) == 64 * 64
        assert len(s2.loc[10000:10004]) == 5 * 64

        sel = f.select_trains(by_index[5:10])
        s3 = sel.get_series('SPB_DET_AGIPD1M-1/DET/7CH0:xtdf', 'image.pulseId')
        assert isinstance(s3, pd.Series)
        assert not s3.index.is_unique
        assert len(s3) == 5 * 64
        np.testing.assert_array_equal(
            s3.index.values, np.arange(10005, 10010).repeat(64)
        )


def test_run_get_series_control(mock_fxe_raw_run):
    run = RunDirectory(mock_fxe_raw_run)
    s = run.get_series('SA1_XTD2_XGM/DOOCS/MAIN', "beamPosition.iyPos.value")
    assert isinstance(s, pd.Series)
    assert len(s) == 480
    assert list(s.index) == list(range(10000, 10480))


def test_run_get_series_select_trains(mock_fxe_raw_run):
    run = RunDirectory(mock_fxe_raw_run)
    sel = run.select_trains(by_id[10100:10150])
    s = sel.get_series('SA1_XTD2_XGM/DOOCS/MAIN', "beamPosition.iyPos.value")
    assert isinstance(s, pd.Series)
    assert len(s) == 50
    assert list(s.index) == list(range(10100, 10150))


def test_run_get_dataframe(mock_fxe_raw_run):
    run = RunDirectory(mock_fxe_raw_run)
    df = run.get_dataframe(fields=[("*_XGM/*", "*.i[xy]Pos*")])
    assert len(df.columns) == 4
    assert "SA1_XTD2_XGM/DOOCS/MAIN/beamPosition.ixPos" in df.columns

    df2 = run.get_dataframe(fields=[("*_XGM/*", "*.i[xy]Pos*")], timestamps=True)
    assert len(df2.columns) == 8
    assert "SA1_XTD2_XGM/DOOCS/MAIN/beamPosition.ixPos" in df2.columns
    assert "SA1_XTD2_XGM/DOOCS/MAIN/beamPosition.ixPos.timestamp" in df2.columns


def test_file_get_array(mock_fxe_control_data):
    with H5File(mock_fxe_control_data) as f:
        arr = f.get_array('FXE_XAD_GEC/CAM/CAMERA:daqOutput', 'data.image.pixels')

    assert isinstance(arr, DataArray)
    assert arr.dims == ('trainId', 'dim_0', 'dim_1')
    assert arr.shape == (400, 255, 1024)
    assert arr.coords['trainId'][0] == 10000


def test_file_get_array_missing_trains(mock_sa3_control_data):
    with H5File(mock_sa3_control_data) as f:
        sel = f.select_trains(by_index[:6])
        arr = sel.get_array(
            'SA3_XTD10_IMGFEL/CAM/BEAMVIEW2:daqOutput', 'data.image.dims'
        )

    assert isinstance(arr, DataArray)
    assert arr.dims == ('trainId', 'dim_0')
    assert arr.shape == (3, 2)
    np.testing.assert_array_less(arr.coords['trainId'], 10006)
    np.testing.assert_array_less(10000, arr.coords['trainId'])


def test_file_get_array_control_roi(mock_sa3_control_data):
    with H5File(mock_sa3_control_data) as f:
        sel = f.select_trains(by_index[:6])
        arr = sel.get_array(
            'SA3_XTD10_VAC/DCTRL/D6_APERT_IN_OK',
            'interlock.a1.AActCommand.value',
            roi=by_index[:25],
        )

    assert isinstance(arr, DataArray)
    assert arr.shape == (6, 25)
    assert arr.coords['trainId'][0] == 10000


@pytest.mark.parametrize('name_in, name_out', [
    (None, 'SA1_XTD2_XGM/DOOCS/MAIN:output.data.intensityTD'),
    ('SA1_XGM', 'SA1_XGM')
], ids=['defaultName', 'explicitName'])
def test_run_get_array(mock_fxe_raw_run, name_in, name_out):
    run = RunDirectory(mock_fxe_raw_run)
    arr = run.get_array(
        'SA1_XTD2_XGM/DOOCS/MAIN:output', 'data.intensityTD',
        extra_dims=['pulse'], name=name_in
    )

    assert isinstance(arr, DataArray)
    assert arr.dims == ('trainId', 'pulse')
    assert arr.shape == (480, 1000)
    assert arr.coords['trainId'][0] == 10000
    assert arr.name == name_out


def test_run_get_array_empty(mock_fxe_raw_run):
    run = RunDirectory(mock_fxe_raw_run)
    arr = run.get_array('FXE_XAD_GEC/CAM/CAMERA_NODATA:daqOutput', 'data.image.pixels')

    assert isinstance(arr, DataArray)
    assert arr.dims[0] == 'trainId'
    assert arr.shape == (0, 255, 1024)


def test_run_get_array_error(mock_fxe_raw_run):
    run = RunDirectory(mock_fxe_raw_run)

    with pytest.raises(SourceNameError):
        run.get_array('bad_name', 'data.intensityTD')

    with pytest.raises(PropertyNameError):
        run.get_array('SA1_XTD2_XGM/DOOCS/MAIN:output', 'bad_name')


def test_run_get_array_select_trains(mock_fxe_raw_run):
    run = RunDirectory(mock_fxe_raw_run)
    sel = run.select_trains(by_id[10100:10150])
    arr = sel.get_array(
        'SA1_XTD2_XGM/DOOCS/MAIN:output', 'data.intensityTD', extra_dims=['pulse']
    )

    assert isinstance(arr, DataArray)
    assert arr.dims == ('trainId', 'pulse')
    assert arr.shape == (50, 1000)
    assert arr.coords['trainId'][0] == 10100


def test_run_get_array_roi(mock_fxe_raw_run):
    run = RunDirectory(mock_fxe_raw_run)
    arr = run.get_array('SA1_XTD2_XGM/DOOCS/MAIN:output', 'data.intensityTD',
                        extra_dims=['pulse'], roi=by_index[:16])

    assert isinstance(arr, DataArray)
    assert arr.dims == ('trainId', 'pulse')
    assert arr.shape == (480, 16)
    assert arr.coords['trainId'][0] == 10000


def test_run_get_array_multiple_per_train(mock_fxe_raw_run):
    run = RunDirectory(mock_fxe_raw_run)
    sel = run.select_trains(np.s_[:2])
    arr = sel.get_array(
        'FXE_DET_LPD1M-1/DET/6CH0:xtdf', 'image.data', roi=np.s_[:, 10:20, 20:40]
    )
    assert isinstance(arr, DataArray)
    assert arr.shape == (256, 1, 10, 20)
    np.testing.assert_array_equal(arr.coords['trainId'], np.repeat([10000, 10001], 128))


def test_run_get_virtual_dataset(mock_fxe_raw_run):
    run = RunDirectory(mock_fxe_raw_run)
    ds = run.get_virtual_dataset('FXE_DET_LPD1M-1/DET/6CH0:xtdf', 'image.data')
    assert isinstance(ds, h5py.Dataset)
    assert ds.is_virtual
    assert ds.shape == (61440, 1, 256, 256)

    # Across two sequence files
    ds = run.get_virtual_dataset(
        'FXE_XAD_GEC/CAM/CAMERA:daqOutput', 'data.image.pixels'
    )
    assert isinstance(ds, h5py.Dataset)
    assert ds.is_virtual
    assert ds.shape == (480, 255, 1024)


def test_run_get_virtual_dataset_filename(mock_fxe_raw_run, tmpdir):
    run = RunDirectory(mock_fxe_raw_run)
    path = str(tmpdir / 'test-vds.h5')
    ds = run.get_virtual_dataset(
        'FXE_DET_LPD1M-1/DET/6CH0:xtdf', 'image.data', filename=path
    )
    assert_isfile(path)
    assert ds.file.filename == path
    assert isinstance(ds, h5py.Dataset)
    assert ds.is_virtual
    assert ds.shape == (61440, 1, 256, 256)


def test_run_get_dask_array(mock_fxe_raw_run):
    import dask.array as da
    run = RunDirectory(mock_fxe_raw_run)
    arr = run.get_dask_array(
        'SA1_XTD2_XGM/DOOCS/MAIN:output', 'data.intensityTD',
    )

    assert isinstance(arr, da.Array)
    assert arr.shape == (480, 1000)
    assert arr.dtype == np.float32


def test_run_get_dask_array_labelled(mock_fxe_raw_run):
    import dask.array as da
    run = RunDirectory(mock_fxe_raw_run)
    arr = run.get_dask_array(
        'SA1_XTD2_XGM/DOOCS/MAIN:output', 'data.intensityTD', labelled=True
    )

    assert isinstance(arr, DataArray)
    assert isinstance(arr.data, da.Array)
    assert arr.dims == ('trainId', 'dim_0')
    assert arr.shape == (480, 1000)
    assert arr.coords['trainId'][0] == 10000


def test_select(mock_fxe_raw_run):
    run = RunDirectory(mock_fxe_raw_run)

    assert 'SPB_XTD9_XGM/DOOCS/MAIN' in run.control_sources

    # Basic selection machinery, glob API
    sel = run.select('*/DET/*', 'image.pulseId')
    assert 'SPB_XTD9_XGM/DOOCS/MAIN' not in sel.control_sources
    assert 'FXE_DET_LPD1M-1/DET/0CH0:xtdf' in sel.instrument_sources
    _, data = sel.train_from_id(10000)
    for source, source_data in data.items():
        assert set(source_data.keys()) == {'image.pulseId', 'metadata'}

    sel_by_list = run.select([
        ('*/DET/*', 'image.pulseId'),
        'FXE_XAD_GEC/CAM/*',
    ])
    assert 'SPB_XTD9_XGM/DOOCS/MAIN' not in sel_by_list.control_sources
    assert 'FXE_DET_LPD1M-1/DET/0CH0:xtdf' in sel_by_list.instrument_sources
    assert sel_by_list['FXE_DET_LPD1M-1/DET/0CH0:xtdf'].keys() == {'image.pulseId'}
    cam_src = 'FXE_XAD_GEC/CAM/CAMERA_NODATA'
    assert cam_src in sel_by_list.control_sources
    assert f'{cam_src}:daqOutput' in sel_by_list.instrument_sources
    assert sel_by_list[cam_src].keys() == run[cam_src].keys()
    assert sel_by_list[f'{cam_src}:daqOutput'].keys() == run[f'{cam_src}:daqOutput'].keys()

    # Basic selection machinery, dict-based API
    sel_by_dict = run.select({
        'SA1_XTD2_XGM/DOOCS/MAIN': None,
        'FXE_DET_LPD1M-1/DET/0CH0:xtdf': {'image.pulseId'}
    })
    assert sel_by_dict.control_sources == {'SA1_XTD2_XGM/DOOCS/MAIN'}
    assert sel_by_dict.instrument_sources == {'FXE_DET_LPD1M-1/DET/0CH0:xtdf'}
    assert sel_by_dict.keys_for_source('FXE_DET_LPD1M-1/DET/0CH0:xtdf') == \
        sel.keys_for_source('FXE_DET_LPD1M-1/DET/0CH0:xtdf')

    # Re-select using * selection, should yield the same keys.
    assert sel.keys_for_source('FXE_DET_LPD1M-1/DET/0CH0:xtdf') == \
        sel.select('FXE_DET_LPD1M-1/DET/0CH0:xtdf', '*') \
           .keys_for_source('FXE_DET_LPD1M-1/DET/0CH0:xtdf')
    assert sel.keys_for_source('FXE_DET_LPD1M-1/DET/0CH0:xtdf') == \
        sel.select({'FXE_DET_LPD1M-1/DET/0CH0:xtdf': {}}) \
           .keys_for_source('FXE_DET_LPD1M-1/DET/0CH0:xtdf')

    # Re-select a different but originally valid key, should fail.
    with pytest.raises(ValueError):
        # ValueError due to globbing.
        sel.select('FXE_DET_LPD1M-1/DET/0CH0:xtdf', 'image.trainId')

    with pytest.raises(PropertyNameError):
        # PropertyNameError via explicit key.
        sel.select({'FXE_DET_LPD1M-1/DET/0CH0:xtdf': {'image.trainId'}})

    # Select by another DataCollection.
    sel_by_dc = run.select(sel)
    assert sel_by_dc.control_sources == sel.control_sources
    assert sel_by_dc.instrument_sources == sel.instrument_sources
    assert sel_by_dc.train_ids == sel.train_ids


@pytest.mark.parametrize(
    'select_str',
    ['*/BEAMVIEW2:daqOutput', '*/BEAMVIEW2*', '*', [('*/BEAMVIEW2:*', 'data.image.*')]]
)
def test_select_require_all(mock_sa3_control_data, select_str):
    # De-select two sources in this example set which have no trains
    # at all as well as one other with partuial trains, to allow
    # matching trains across all sources with the same result.
    run = H5File(mock_sa3_control_data) \
        .deselect([('SA3_XTD10_MCP/ADC/1:*', '*'),
                   ('SA3_XTD10_IMGFEL/CAM/BEAMVIEW:*', '*'),
                   ('SA3_XTD10_IMGFEL/CAM/BEAMVIEW3:*', '*')])
    subrun = run.select(select_str, require_all=True)
    np.testing.assert_array_equal(subrun.train_ids, run.train_ids[1::2])

    # The train IDs are held by ndarrays during this operation, make
    # sure it's a list of np.uint64 again.
    assert isinstance(subrun.train_ids, list)
    assert all([isinstance(x, np.uint64) for x in subrun.train_ids])


def test_select_require_any(mock_sa3_control_data):
    run = H5File(mock_sa3_control_data)

    # BEAMVIEW2 has 250/500 trains, BEAMVIEW3 has 200/500 trains.
    # Compare the train IDs resulting from a require-any select with the
    # union of their respective train IDs.
    np.testing.assert_array_equal(
        run.select('*/BEAMVIEW*:daqOutput', require_any=True).train_ids,
        np.union1d(
            run.select('*/BEAMVIEW2:daqOutput', require_all=True).train_ids,
            run.select('*/BEAMVIEW3:daqOutput', require_all=True).train_ids
        ))

    # BEAMVIEW has no trains, should also yield an empty list.
    assert run.select('*/BEAMVIEW:daqOutput', require_any=True).train_ids == []


def test_deselect(mock_fxe_raw_run):
    run = RunDirectory(mock_fxe_raw_run)

    xtd9_xgm = 'SPB_XTD9_XGM/DOOCS/MAIN'
    assert xtd9_xgm in run.control_sources

    sel = run.deselect('*_XGM/DOOCS*')
    assert xtd9_xgm not in sel.control_sources
    assert 'FXE_DET_LPD1M-1/DET/0CH0:xtdf' in sel.instrument_sources

    sel = run.deselect('*_XGM/DOOCS*', '*.ixPos')
    assert xtd9_xgm in sel.control_sources
    assert 'beamPosition.ixPos.value' not in sel.selection[xtd9_xgm]
    assert 'beamPosition.iyPos.value' in sel.selection[xtd9_xgm]

    sel = run.deselect(run.select('*_XGM/DOOCS*'))
    assert xtd9_xgm not in sel.control_sources
    assert 'FXE_DET_LPD1M-1/DET/0CH0:xtdf' in sel.instrument_sources


def test_select_trains(mock_fxe_raw_run):
    run = RunDirectory(mock_fxe_raw_run)

    assert len(run.train_ids) == 480

    sel = run.select_trains(by_id[10200:10220])
    assert sel.train_ids == list(range(10200, 10220))

    sel = run.select_trains(by_index[:10])
    assert sel.train_ids == list(range(10000, 10010))

    with catch_warnings(record=True) as w:
        sel = run.select_trains(by_id[9000:9100])  # Before data
        assert sel.train_ids == []
        assert len(w) == 1
        assert "before" in str(w[0].message)

    with catch_warnings(record=True) as w:
        sel = run.select_trains(by_id[12000:12500])  # After data
        assert sel.train_ids == []
        assert len(w) == 1
        assert "after" in str(w[0].message)

    # Select a list of train IDs
    sel = run.select_trains(by_id[[9950, 10000, 10101, 10500]])
    assert sel.train_ids == [10000, 10101]

    with catch_warnings(record=True) as w:
        sel = run.select_trains(by_id[[9900, 10600]])
        assert sel.train_ids == []
        assert len(w) == 1
        assert "not found" in str(w[0].message)

    # Select a list of indexes
    sel = run.select_trains(by_index[[5, 25]])
    assert sel.train_ids == [10005, 10025]

    with pytest.raises(IndexError):
        run.select_trains(by_index[[480]])


def test_split_trains(mock_fxe_raw_run):
    run = RunDirectory(mock_fxe_raw_run)
    assert len(run.train_ids) == 480

    chunks = list(run.split_trains(3))
    assert len(chunks) == 3
    assert {len(c.train_ids) for c in chunks} == {160}

    chunks = list(run.split_trains(4, trains_per_part=100))
    assert len(chunks) == 5
    assert {len(c.train_ids) for c in chunks} == {96}


def test_train_timestamps(mock_scs_run):
    run = RunDirectory(mock_scs_run)

    tss = run.train_timestamps(labelled=False)
    assert isinstance(tss, np.ndarray)
    assert tss.shape == (len(run.train_ids),)
    assert tss.dtype == np.dtype('datetime64[ns]')
    assert np.all(np.diff(tss).astype(np.uint64) > 0)

    # Convert numpy datetime64[ns] to Python datetime (dropping some precision)
    dt0 = tss[0].astype('datetime64[ms]').item().replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    assert dt0 > (now - timedelta(days=1))  # assuming tests take < 1 day to run
    assert dt0 < now

    tss_ser = run.train_timestamps(labelled=True)
    assert isinstance(tss_ser, pd.Series)
    np.testing.assert_array_equal(tss_ser.values, tss)
    np.testing.assert_array_equal(tss_ser.index, run.train_ids)


def test_train_timestamps_nat(mock_fxe_control_data):
    f = H5File(mock_fxe_control_data)
    tss = f.train_timestamps()
    assert tss.shape == (len(f.train_ids),)
    if f.files[0].format_version == '0.5':
        assert np.all(np.isnat(tss))
    else:
        assert not np.any(np.isnat(tss))


def test_union(mock_fxe_raw_run):
    run = RunDirectory(mock_fxe_raw_run)

    xgm = "SPB_XTD9_XGM/DOOCS/MAIN"
    camera = "FXE_XAD_GEC/CAM/CAMERA"

    # Test union of different sources
    sel1 = run.select(xgm, 'beamPosition.ixPos')
    sel2 = run.select(xgm, 'beamPosition.iyPos')
    joined = sel1.union(sel2)
    assert joined.control_sources == { xgm }
    assert joined.selection == {
        xgm: {
            'beamPosition.ixPos.value',
            'beamPosition.iyPos.value',
        }
    }

    # Test union of different train selections
    sel1 = run.select_trains(by_id[10200:10220])
    sel2 = run.select_trains(by_index[:10])
    joined = sel1.union(sel2)
    assert joined.train_ids == list(range(10000, 10010)) + list(range(10200, 10220))

    # Test union of different sources in different train selections
    sel1 = run.select(xgm).select_trains(by_index[:5])
    sel2 = run.select(camera).select_trains(by_index[-5:])
    joined = sel1.union(sel2)
    expected_tids = run.train_ids[:5] + run.train_ids[-5:]

    assert joined.train_ids == expected_tids
    assert joined[xgm].train_ids == expected_tids
    assert joined[camera].train_ids == expected_tids


def test_union_raw_proc(mock_spb_raw_run, mock_spb_proc_run):
    raw_run = RunDirectory(mock_spb_raw_run)
    proc_run = RunDirectory(mock_spb_proc_run)
    run = raw_run.deselect('*AGIPD1M*').union(proc_run)

    assert run.all_sources == (raw_run.all_sources | proc_run.all_sources)
    if raw_run.run_metadata()['dataFormatVersion'] != '0.5':
        assert run.is_single_run


def test_union_multiple_runs(mock_spb_raw_run, mock_jungfrau_run, mock_scs_run):
    run_spb = RunDirectory(mock_spb_raw_run)
    run_jf = RunDirectory(mock_jungfrau_run)
    run_scs = RunDirectory(mock_scs_run)

    assert run_spb.is_single_run
    assert run_jf.is_single_run
    assert run_scs.is_single_run

    # Union in one go
    u1 = run_spb.union(run_jf, run_scs)
    assert u1.all_sources == (run_spb.all_sources | run_jf.all_sources | run_scs.all_sources)
    assert not u1.is_single_run

    # Union in two steps
    u2 = run_scs.union(run_jf).union(run_spb)
    assert u2.all_sources == u1.all_sources
    assert not u1.is_single_run


def test_read_skip_invalid(mock_lpd_data, empty_h5_file, capsys):
    d = DataCollection.from_paths([mock_lpd_data, empty_h5_file])
    assert d.instrument_sources == {'FXE_DET_LPD1M-1/DET/0CH0:xtdf'}
    out, err = capsys.readouterr()
    assert "Skipping file" in err


def test_run_immutable_sources(mock_fxe_raw_run):
    test_run = RunDirectory(mock_fxe_raw_run)
    before = len(test_run.all_sources)

    with pytest.raises(AttributeError):
        test_run.all_sources.pop()

    assert len(test_run.all_sources) == before


def test_open_run(mock_spb_raw_and_proc_run):
    mock_data_root, raw_run_dir, proc_run_dir = mock_spb_raw_and_proc_run

    with mock.patch('extra_data.read_machinery.DATA_ROOT_DIR', mock_data_root):
        # With integers
        run = open_run(proposal=2012, run=238)
        paths = {f.filename for f in run.files}

        assert paths
        for path in paths:
            assert '/raw/' in path

        # With strings
        run = open_run(proposal='2012', run='238')
        assert {f.filename for f in run.files} == paths

        # With numpy integers
        run = open_run(proposal=np.int64(2012), run=np.uint16(238))
        assert {f.filename for f in run.files} == paths

        # Proc folder
        proc_run = open_run(proposal=2012, run=238, data='proc')

        proc_paths = {f.filename for f in proc_run.files}
        assert proc_paths
        for path in proc_paths:
            assert '/raw/' not in path

        # All folders
        all_run = open_run(proposal=2012, run=238, data='all')

        # Raw contains all sources.
        assert run.all_sources == all_run.all_sources

        # Proc is a true subset.
        assert proc_run.all_sources < all_run.all_sources

        for source, srcdata in all_run._sources_data.items():
            for file in srcdata.files:
                if '/DET/' in source:
                    # AGIPD data is in proc.
                    assert '/raw/' not in file.filename
                else:
                    # Non-AGIPD data is in raw.
                    # (CAM, XGM)
                    assert '/proc/' not in file.filename

        # Delete the proc data
        shutil.rmtree(proc_run_dir)
        assert not os.path.isdir(proc_run_dir)

        with catch_warnings(record=True) as w:
            # Opening a run with 'all', with no proc data
            all_run = open_run(proposal=2012, run=238, data='all')

            # Attempting to open the proc data should raise a warning
            assert len(w) == 1

        # It should have opened at least the raw data
        assert run.all_sources == all_run.all_sources

        # Run that doesn't exist
        with pytest.raises(Exception):
            open_run(proposal=2012, run=999)

        # run directory exists but contains no data
        os.makedirs(proc_run_dir)
        with catch_warnings(record=True) as w:
            open_run(proposal=2012, run=238, data='all')
            assert len(w) == 1

        # Helper function to write an alias file at a specific path
        def write_aliases(path):
            aliases_path.parent.mkdir(parents=True, exist_ok=True)
            aliases_path.write_text(dedent("""
            xgm: SA1_XTD2_XGM/DOOCS/MAIN
            """))

        # To set the aliases, we should be able to use a string relative to the
        # proposal directory.
        aliases_path = Path(mock_data_root) / "SPB/201830/p002012/foo.yml"
        write_aliases(aliases_path)
        run = open_run(2012, 238, data="all", aliases="{}/foo.yml")
        assert "xgm" in run.alias

        # And a proper path
        aliases_path = Path(mock_data_root) / "foo.yml"
        write_aliases(aliases_path)
        run = open_run(2012, 238, aliases=aliases_path)
        assert "xgm" in run.alias

        # And a plain string
        run = open_run(2012, 238, aliases=str(aliases_path))
        assert "xgm" in run.alias

        # If the default file exists, it should be used automatically
        aliases_path = Path(DEFAULT_ALIASES_FILE.format(mock_data_root + "/SPB/201830/p002012"))
        write_aliases(aliases_path)
        run = open_run(2012, 238)
        assert "xgm" in run.alias

def test_open_file(mock_sa3_control_data):
    f = H5File(mock_sa3_control_data)
    file_access = f.files[0]
    assert file_access.format_version in ('0.5', '1.0', '1.2')
    assert 'SA3_XTD10_VAC/TSENS/S30180K' in f.control_sources
    if file_access.format_version == '0.5':
        assert 'METADATA/dataSourceId' in file_access.file
    else:
        assert 'METADATA/dataSources/dataSourceId' in file_access.file


def open_run_daemonized_helper(mock_data_root):
    with mock.patch('extra_data.read_machinery.DATA_ROOT_DIR', mock_data_root):
        open_run(2012, 238, data="all", parallelize=False)

def test_open_run_daemonized(mock_spb_raw_and_proc_run):
    mock_data_root, raw_run_dir, proc_run_dir = mock_spb_raw_and_proc_run

    # Daemon processes can't start their own children, check that opening a run is still possible.
    p = Process(target=open_run_daemonized_helper, args=(mock_data_root,), daemon=True)
    p.start()
    p.join()

    assert p.exitcode == 0

@pytest.mark.skipif(hasattr(os, 'geteuid') and os.geteuid() == 0,
                    reason="cannot run permission tests as root")
def test_permission():
    d = mkdtemp()
    os.chmod(d, not stat.S_IRUSR)
    with pytest.raises(PermissionError) as excinfo:
        run = RunDirectory(d)
    assert "Permission denied" in str(excinfo.value)
    assert d in str(excinfo.value)


def test_empty_file_info(mock_empty_file, capsys):
    f = H5File(mock_empty_file)
    f.info()  # smoke test


def test_get_data_counts(mock_spb_raw_run):
    run = RunDirectory(mock_spb_raw_run)
    count = run.get_data_counts('SPB_XTD9_XGM/DOOCS/MAIN', 'beamPosition.ixPos.value')
    assert count.index.tolist() == run.train_ids
    assert (count.values == 1).all()


def test_get_run_value(mock_fxe_control_data):
    f = H5File(mock_fxe_control_data)
    src = 'FXE_XAD_GEC/CAM/CAMERA'
    val = f.get_run_value(src, 'firmwareVersion')
    assert isinstance(val, np.int32)
    assert f.get_run_value(src, 'firmwareVersion.value') == val

    with pytest.raises(SourceNameError):
        f.get_run_value(src + '_NONEXIST', 'firmwareVersion')

    with pytest.raises(PropertyNameError):
        f.get_run_value(src, 'non.existant')


def test_get_run_value_union_multirun(mock_fxe_control_data, mock_fxe_control_data1):
    f = H5File(mock_fxe_control_data)
    f2 = H5File(mock_fxe_control_data1)
    data = f.union(f2)
    with pytest.raises(MultiRunError):
        data.run_metadata()

    with pytest.raises(MultiRunError):
        data.get_run_value('FXE_XAD_GEC/CAM/CAMERA', 'firmwareVersion')

    with pytest.raises(MultiRunError):
        data.get_run_values('FXE_XAD_GEC/CAM/CAMERA')


def test_get_run_value_union(mock_fxe_control_data, mock_sa3_control_data):
    f = H5File(mock_fxe_control_data)
    f2 = H5File(mock_sa3_control_data)
    data = f.union(f2)
    if data.files[0].format_version != '0.5':
        assert data.get_run_value(
            'FXE_XAD_GEC/CAM/CAMERA', 'firmwareVersion') == 0

        assert (
            data.run_metadata()["runNumber"] ==
            f.run_metadata()["runNumber"] ==
            f2.run_metadata()["runNumber"]
        )


def test_get_run_values(mock_fxe_control_data):
    f = H5File(mock_fxe_control_data)
    src = 'FXE_XAD_GEC/CAM/CAMERA'
    d = f.get_run_values(src, )
    assert isinstance(d['firmwareVersion.value'], np.int32)
    assert isinstance(d['enableShutter.value'], np.uint8)


def test_get_run_values_no_trains(mock_jungfrau_run):
    run = RunDirectory(mock_jungfrau_run)
    sel = run.select_trains(np.s_[:0])
    d = sel.get_run_values('SPB_IRDA_JF4M/MDL/POWER')
    assert isinstance(d['voltage.value'], np.float64)


def test_inspect_key_no_trains(mock_jungfrau_run):
    run = RunDirectory(mock_jungfrau_run)
    sel = run.select_trains(np.s_[:0])

    # CONTROL
    jf_pwr_voltage = sel['SPB_IRDA_JF4M/MDL/POWER', 'voltage']
    assert jf_pwr_voltage.shape == (0,)
    assert jf_pwr_voltage.dtype == np.dtype(np.float64)

    # INSTRUMENT
    jf_m1_data = sel['SPB_IRDA_JF4M/DET/JNGFR01:daqOutput', 'data.adc']
    assert jf_m1_data.shape == (0, 16, 512, 1024)
    assert jf_m1_data.dtype == np.dtype(np.uint16)


def test_run_metadata(mock_spb_raw_run):
    run = RunDirectory(mock_spb_raw_run)
    md = run.run_metadata()
    if run.files[0].format_version == '0.5':
        assert md == {'dataFormatVersion': '0.5'}
    else:
        assert md['dataFormatVersion'] in ('1.0', '1.2')
        assert set(md) == {
            'dataFormatVersion', 'creationDate', 'updateDate', 'daqLibrary',
            'karaboFramework', 'proposalNumber', 'runNumber', 'runType',
            'sample', 'sequenceNumber',
        }
        assert isinstance(md['creationDate'], str)


def test_run_metadata_no_trains(mock_scs_run):
    run = RunDirectory(mock_scs_run)
    sel = run.select_trains(np.s_[:0])
    md = sel.run_metadata()
    assert md['dataFormatVersion'] == '1.0'
