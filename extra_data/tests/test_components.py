import dask.array as da
import h5py
import numpy as np
import os.path as osp
import pytest
from testpath import assert_isfile

from extra_data.reader import RunDirectory, H5File, by_id, by_index
from extra_data.components import AGIPD1M, LPD1M, identify_multimod_detectors


def test_get_array(mock_fxe_raw_run):
    run = RunDirectory(mock_fxe_raw_run)
    det = LPD1M(run.select_trains(by_index[:3]))
    assert det.detector_name == 'FXE_DET_LPD1M-1'

    arr = det.get_array('image.data')
    assert arr.shape == (16, 3, 128, 256, 256)
    assert arr.dims == ('module', 'train', 'pulse', 'slow_scan', 'fast_scan')

    arr = det.get_array('image.data', pulses=by_index[:10], unstack_pulses=False)
    assert arr.shape == (16, 30, 256, 256)
    assert arr.dims == ('module', 'train_pulse', 'slow_scan', 'fast_scan')

def test_get_array_pulse_id(mock_fxe_raw_run):
    run = RunDirectory(mock_fxe_raw_run)
    det = LPD1M(run.select_trains(by_index[:3]))
    arr = det.get_array('image.data', pulses=by_id[0])
    assert arr.shape == (16, 3, 1, 256, 256)
    assert (arr.coords['pulse'] == 0).all()

    arr = det.get_array('image.data', pulses=by_id[:5])
    assert arr.shape == (16, 3, 5, 256, 256)

    # Empty selection
    arr = det.get_array('image.data', pulses=by_id[:0])
    assert arr.shape == (16, 0, 0, 256, 256)

    arr = det.get_array('image.data', pulses=by_id[122:])
    assert arr.shape == (16, 3, 6, 256, 256)

    arr = det.get_array('image.data', pulses=by_id[[1, 7, 22, 23]])
    assert arr.shape == (16, 3, 4, 256, 256)
    assert list(arr.coords['pulse']) == [1, 7, 22, 23]


def test_get_array_pulse_indexes(mock_fxe_raw_run):
    run = RunDirectory(mock_fxe_raw_run)
    det = LPD1M(run.select_trains(by_index[:3]))
    arr = det.get_array('image.data', pulses=by_index[0])
    assert arr.shape == (16, 3, 1, 256, 256)
    assert (arr.coords['pulse'] == 0).all()

    arr = det.get_array('image.data', pulses=by_index[:5])
    assert arr.shape == (16, 3, 5, 256, 256)

    # Empty selection
    arr = det.get_array('image.data', pulses=by_index[:0])
    assert arr.shape == (16, 0, 0, 256, 256)

    arr = det.get_array('image.data', pulses=by_index[122:])
    assert arr.shape == (16, 3, 6, 256, 256)

    arr = det.get_array('image.data', pulses=by_index[[1, 7, 22, 23]])
    assert arr.shape == (16, 3, 4, 256, 256)


def test_get_array_pulse_id_reduced_data(mock_reduced_spb_proc_run):
    run = RunDirectory(mock_reduced_spb_proc_run)
    det = AGIPD1M(run.select_trains(by_index[:3]))
    arr = det.get_array('image.data', pulses=by_id[0])
    assert arr.shape == (16, 3, 1, 512, 128)
    assert (arr.coords['pulse'] == 0).all()

    arr = det.get_array('image.data', pulses=by_id[:5])
    assert (arr.coords['pulse'] < 5).all()

    # Empty selection
    arr = det.get_array('image.data', pulses=by_id[:0])
    assert arr.shape == (16, 0, 0, 512, 128)

    arr = det.get_array('image.data', pulses=by_id[5:])
    assert (arr.coords['pulse'] >= 5).all()

    arr = det.get_array('image.data', pulses=by_id[[1, 7, 15, 23]])
    assert np.isin(arr.coords['pulse'], [1, 7, 15, 23]).all()


def test_get_array_pulse_indexes_reduced_data(mock_reduced_spb_proc_run):
    run = RunDirectory(mock_reduced_spb_proc_run)
    det = AGIPD1M(run.select_trains(by_index[:3]))
    arr = det.get_array('image.data', pulses=by_index[0])
    assert arr.shape == (16, 3, 1, 512, 128)
    assert (arr.coords['pulse'] == 0).all()

    arr = det.get_array('image.data', pulses=by_index[:5])
    assert (arr.coords['pulse'] < 5).all()

    # Empty selection
    arr = det.get_array('image.data', pulses=by_index[:0])
    assert arr.shape == (16, 0, 0, 512, 128)

    arr = det.get_array('image.data', pulses=np.s_[5:])
    assert (arr.coords['pulse'] >= 5).all()

    arr = det.get_array('image.data', pulses=by_index[[1, 7, 15, 23]])
    assert np.isin(arr.coords['pulse'], [1, 7, 15, 23]).all()

    arr = det.get_array('image.data', pulses=[1, 7, 15, 23])
    assert np.isin(arr.coords['pulse'], [1, 7, 15, 23]).all()

def test_get_dask_array(mock_fxe_raw_run):
    run = RunDirectory(mock_fxe_raw_run)
    det = LPD1M(run)
    arr = det.get_dask_array('image.data')

    assert isinstance(arr.data, da.Array)
    assert arr.shape == (16, 480 * 128, 1, 256, 256)
    assert arr.dims == ('module', 'train_pulse', 'dim_0', 'dim_1', 'dim_2')
    np.testing.assert_array_equal(arr.coords['module'], np.arange(16))
    np.testing.assert_array_equal(
        arr.coords['trainId'], np.repeat(np.arange(10000, 10480), 128)
    )
    np.testing.assert_array_equal(
        arr.coords['pulseId'], np.tile(np.arange(0, 128), 480)
    )

    arr_cellid = det.get_dask_array('image.data', subtrain_index='cellId')
    assert arr_cellid.coords['cellId'].shape == (480 * 128,)


def test_get_dask_array_reduced_data(mock_reduced_spb_proc_run):
    run = RunDirectory(mock_reduced_spb_proc_run)
    det = AGIPD1M(run)
    arr = det.get_dask_array('image.data')

    assert arr.shape[2:] == (512, 128)
    assert arr.dims == ('module', 'train_pulse', 'dim_0', 'dim_1')
    np.testing.assert_array_equal(arr.coords['module'], np.arange(16))
    assert np.isin(arr.coords['trainId'], np.arange(10000, 10480)).all()
    assert np.isin(arr.coords['pulseId'], np.arange(0, 20)).all()


def test_iterate(mock_fxe_raw_run):
    run = RunDirectory(mock_fxe_raw_run)
    det = LPD1M(run.select_trains(by_index[:2]))
    it = iter(det.trains())
    tid, d = next(it)
    assert d['image.data'].shape == (16, 1, 128, 256, 256)
    assert d['image.data'].dims == ('module', 'train', 'pulse', 'slow_scan', 'fast_scan')

    tid, d = next(it)
    assert d['image.data'].shape == (16, 1, 128, 256, 256)

    with pytest.raises(StopIteration):
        next(it)


def test_iterate_pulse_id(mock_fxe_raw_run):
    run = RunDirectory(mock_fxe_raw_run)
    det = LPD1M(run.select_trains(by_index[:3]))
    tid, d = next(iter(det.trains(pulses=by_id[0])))
    assert d['image.data'].shape == (16, 1, 1, 256, 256)

    tid, d = next(iter(det.trains(pulses=by_id[:5])))
    assert d['image.data'].shape == (16, 1, 5, 256, 256)

    tid, d = next(iter(det.trains(pulses=by_id[122:])))
    assert d['image.data'].shape == (16, 1, 6, 256, 256)

    tid, d = next(iter(det.trains(pulses=by_id[[1, 7, 22, 23]])))
    assert d['image.data'].shape == (16, 1, 4, 256, 256)
    assert list(d['image.data'].coords['pulse']) == [1, 7, 22, 23]


def test_iterate_pulse_index(mock_fxe_raw_run):
    run = RunDirectory(mock_fxe_raw_run)
    det = LPD1M(run.select_trains(by_index[:3]))
    tid, d = next(iter(det.trains(pulses=by_index[0])))
    assert d['image.data'].shape == (16, 1, 1, 256, 256)

    tid, d = next(iter(det.trains(pulses=by_index[:5])))
    assert d['image.data'].shape == (16, 1, 5, 256, 256)

    tid, d = next(iter(det.trains(pulses=by_index[122:])))
    assert d['image.data'].shape == (16, 1, 6, 256, 256)

    tid, d = next(iter(det.trains(pulses=by_index[[1, 7, 22, 23]])))
    assert d['image.data'].shape == (16, 1, 4, 256, 256)
    assert list(d['image.data'].coords['pulse']) == [1, 7, 22, 23]

def test_write_virtual_cxi(mock_spb_proc_run, tmpdir):
    run = RunDirectory(mock_spb_proc_run)
    det = AGIPD1M(run)

    test_file = osp.join(str(tmpdir), 'test.cxi')
    det.write_virtual_cxi(test_file)
    assert_isfile(test_file)

    with h5py.File(test_file, 'r') as f:
        det_grp = f['entry_1/instrument_1/detector_1']
        ds = det_grp['data']
        assert isinstance(ds, h5py.Dataset)
        assert ds.is_virtual
        assert ds.shape[1:] == (16, 512, 128)
        assert 'axes' in ds.attrs

        assert len(ds.virtual_sources()) == 16

        # Check position of each source file in the modules dimension
        for src in ds.virtual_sources():
            start, _, block, count = src.vspace.get_regular_hyperslab()
            assert block[1] == 1
            assert count[1] == 1

            expected_file = 'CORR-R0238-AGIPD{:0>2}-S00000.h5'.format(start[1])
            assert osp.basename(src.file_name) == expected_file

        # Check presence of other datasets
        assert 'gain' in det_grp
        assert 'mask' in det_grp
        assert 'experiment_identifier' in det_grp

def test_write_virtual_cxi_some_modules(mock_spb_proc_run, tmpdir):
    run = RunDirectory(mock_spb_proc_run)
    det = AGIPD1M(run, modules=[3, 4, 8, 15])

    test_file = osp.join(str(tmpdir), 'test.cxi')
    det.write_virtual_cxi(test_file)
    assert_isfile(test_file)

    with h5py.File(test_file, 'r') as f:
        det_grp = f['entry_1/instrument_1/detector_1']
        ds = det_grp['data']
        assert ds.shape[1:] == (16, 512, 128)

def test_write_virtual_cxi_raw_data(mock_fxe_raw_run, tmpdir, caplog):
    import logging
    caplog.set_level(logging.INFO)
    run = RunDirectory(mock_fxe_raw_run)
    det = LPD1M(run)

    test_file = osp.join(str(tmpdir), 'test.cxi')
    det.write_virtual_cxi(test_file)
    assert_isfile(test_file)

    with h5py.File(test_file, 'r') as f:
        det_grp = f['entry_1/instrument_1/detector_1']
        ds = det_grp['data']
        assert ds.shape[1:] == (16, 1, 256, 256)


def test_write_virtual_cxi_reduced_data(mock_reduced_spb_proc_run, tmpdir):
    run = RunDirectory(mock_reduced_spb_proc_run)
    det = AGIPD1M(run)

    test_file = osp.join(str(tmpdir), 'test.cxi')
    det.write_virtual_cxi(test_file)
    assert_isfile(test_file)

    with h5py.File(test_file, 'r') as f:
        det_grp = f['entry_1/instrument_1/detector_1']
        ds = det_grp['data']
        assert ds.shape[1:] == (16, 512, 128)


def test_write_selected_frames(mock_spb_raw_run, tmp_path):
    run = RunDirectory(mock_spb_raw_run)
    det = AGIPD1M(run)

    trains = np.repeat(np.arange(10000, 10010), 3)
    pulses = np.tile([0, 1, 5], 10)
    test_file = str(tmp_path / 'sel_frames.h5')
    det.write_frames(test_file, trains, pulses)
    assert_isfile(test_file)

    with H5File(test_file) as f:
        np.testing.assert_array_equal(
            f.get_array('SPB_DET_AGIPD1M-1/DET/0CH0:xtdf', 'image.pulseId')[:, 0],
            pulses
        )
        assert f.instrument_sources == {
            f'SPB_DET_AGIPD1M-1/DET/{i}CH0:xtdf' for i in range(16)
        }


def test_write_selected_frames_proc(mock_spb_proc_run, tmp_path):
    run = RunDirectory(mock_spb_proc_run)
    det = AGIPD1M(run)

    trains = np.repeat(np.arange(10000, 10010), 3)
    pulses = np.tile([0, 1, 5], 10)
    test_file = str(tmp_path / 'sel_frames.h5')
    det.write_frames(test_file, trains, pulses)
    assert_isfile(test_file)

    with H5File(test_file) as f:
        np.testing.assert_array_equal(
            f.get_array('SPB_DET_AGIPD1M-1/DET/0CH0:xtdf', 'image.pulseId'),
            pulses
        )
        assert f.instrument_sources == {
            f'SPB_DET_AGIPD1M-1/DET/{i}CH0:xtdf' for i in range(16)
        }


def test_identify_multimod_detectors(mock_fxe_raw_run):
    run = RunDirectory(mock_fxe_raw_run)
    name, cls = identify_multimod_detectors(run, single=True)
    assert name == 'FXE_DET_LPD1M-1'
    assert cls is LPD1M

    dets = identify_multimod_detectors(run, single=False)
    assert dets == {(name, cls)}


def test_identify_multimod_detectors_multi(mock_fxe_raw_run, mock_spb_raw_run):
    fxe_run = RunDirectory(mock_fxe_raw_run)
    spb_run = RunDirectory(mock_spb_raw_run)
    combined = fxe_run.select('*LPD1M*').union(spb_run)

    dets = identify_multimod_detectors(combined, single=False)
    assert dets == {('FXE_DET_LPD1M-1', LPD1M), ('SPB_DET_AGIPD1M-1', AGIPD1M)}

    with pytest.raises(ValueError):
        identify_multimod_detectors(combined, single=True)

    name, cls = identify_multimod_detectors(combined, single=True, clses=[AGIPD1M])
    assert name == 'SPB_DET_AGIPD1M-1'
    assert cls is AGIPD1M
