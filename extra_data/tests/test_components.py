import dask.array as da
import h5py
import numpy as np
import os.path as osp
import pytest
from testpath import assert_isfile

from extra_data.reader import RunDirectory, H5File, by_id, by_index
from extra_data.components import (
    AGIPD1M, DSSC1M, LPD1M, JUNGFRAU, identify_multimod_detectors,
)


def test_get_array(mock_fxe_raw_run):
    run = RunDirectory(mock_fxe_raw_run)
    det = LPD1M(run.select_trains(by_index[:3]))
    assert det.detector_name == 'FXE_DET_LPD1M-1'

    arr = det.get_array('image.data')
    assert arr.dtype == np.uint16
    assert arr.shape == (16, 3, 128, 256, 256)
    assert arr.dims == ('module', 'train', 'pulse', 'slow_scan', 'fast_scan')

    arr = det.get_array('image.data', pulses=by_index[:10], unstack_pulses=False)
    assert arr.shape == (16, 30, 256, 256)
    assert arr.dtype == np.uint16
    assert arr.dims == ('module', 'train_pulse', 'slow_scan', 'fast_scan')

    # fill value
    with pytest.raises(ValueError):
        det.get_array('image.data', fill_value=np.nan)
    
    arr = det.get_array('image.data', astype=np.float32)
    assert arr.dtype == np.float32


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


def test_get_array_with_cell_ids(mock_fxe_raw_run):
    run = RunDirectory(mock_fxe_raw_run)
    det = LPD1M(run.select_trains(by_index[:3]))
    arr = det.get_array('image.data', subtrain_index='cellId')
    assert arr.shape == (16, 3, 128, 256, 256)
    assert arr.dims == ('module', 'train', 'cell', 'slow_scan', 'fast_scan')

    arr = det.get_array('image.data', pulses=by_id[0], subtrain_index='cellId')
    assert arr.shape == (16, 3, 1, 256, 256)
    assert (arr.coords['cell'] == 0).all()


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


def test_get_array_gap(mock_lpd_mini_gap_run):
    run = RunDirectory(mock_lpd_mini_gap_run)
    det = LPD1M(run, modules=[0, 1])

    # All pulses
    arr = det.get_array('image.data')
    assert arr.shape == (2, 5, 10, 256, 256)
    np.testing.assert_array_equal(arr[1, :, 8, 0, 0], [1, 2, 0, 3, 4])

    # Selected pulses
    arr = det.get_array('image.data', pulses=[8])
    assert arr.shape == (2, 5, 1, 256, 256)
    np.testing.assert_array_equal(arr[1, :, 0, 0, 0], [1, 2, 0, 3, 4])


def test_get_array_roi(mock_fxe_raw_run):
    run = RunDirectory(mock_fxe_raw_run)
    det = LPD1M(run.select_trains(by_index[:3]))
    assert det.detector_name == 'FXE_DET_LPD1M-1'

    arr = det.get_array('image.data', roi=np.s_[10:60, 100:200])
    assert arr.shape == (16, 3, 128, 50, 100)
    assert arr.dims == ('module', 'train', 'pulse', 'slow_scan', 'fast_scan')


def test_get_array_roi_dssc(mock_scs_run):
    run = RunDirectory(mock_scs_run)
    det = DSSC1M(run, modules=[3])

    arr = det.get_array('image.data', roi=np.s_[20:25, 40:52])
    assert arr.shape == (1, 128, 64, 5, 12)


def test_ndarray_module_gaps(mock_fxe_raw_run):
    run = RunDirectory(mock_fxe_raw_run)
    det = LPD1M(run, modules=[2]).select_trains(np.s_[:3])
    det_data = det['image.data']
    assert det_data.shape == (1, 128 * 3, 256, 256)
    assert det_data.ndarray().shape == (1, 128 * 3, 256, 256)

    arr_w_gaps = det_data.ndarray(module_gaps=True, fill_value=7)
    assert arr_w_gaps.shape == (16, 128 * 3, 256, 256)
    assert arr_w_gaps[:, 0, 0, 0].tolist() == ([7] * 2) + [0] + ([7] * 13)


def test_get_array_lpd_parallelgain(mock_lpd_parallelgain_run):
    run = RunDirectory(mock_lpd_parallelgain_run)
    det = LPD1M(run.select_trains(by_index[:2]), parallel_gain=True)
    assert det.detector_name == 'FXE_DET_LPD1M-1'

    arr = det.get_array('image.data')
    assert arr.shape == (16, 2, 3, 100, 256, 256)
    assert arr.dims == ('module', 'train', 'gain', 'pulse', 'slow_scan', 'fast_scan')
    np.testing.assert_array_equal(arr.coords['gain'], np.arange(3))
    np.testing.assert_array_equal(arr.coords['pulse'], np.arange(100))


def test_get_array_lpd_parallelgain_select_pulses(mock_lpd_parallelgain_run):
    run = RunDirectory(mock_lpd_parallelgain_run)
    det = LPD1M(run.select_trains(by_index[:2]), parallel_gain=True)
    assert det.detector_name == 'FXE_DET_LPD1M-1'

    arr = det.get_array('image.data', pulses=np.s_[:5])
    assert arr.shape == (16, 2, 3, 5, 256, 256)
    assert arr.dims == ('module', 'train', 'gain', 'pulse', 'slow_scan', 'fast_scan')
    np.testing.assert_array_equal(arr.coords['gain'], np.arange(3))
    np.testing.assert_array_equal(arr.coords['pulse'], np.arange(5))

    arr = det.get_array('image.data', pulses=by_id[:5])
    assert arr.shape == (16, 2, 3, 5, 256, 256)
    np.testing.assert_array_equal(arr.coords['pulse'], np.arange(5))


def test_get_array_jungfrau(mock_jungfrau_run):
    run = RunDirectory(mock_jungfrau_run)
    jf = JUNGFRAU(run.select_trains(by_index[:2]))
    assert jf.detector_name == 'SPB_IRDA_JF4M'

    arr = jf.get_array('data.adc')
    assert arr.shape == (8, 2, 16, 512, 1024)
    assert arr.dims == ('module', 'train', 'cell', 'slow_scan', 'fast_scan')
    np.testing.assert_array_equal(arr.coords['train'], [10000, 10001])

    arr = jf.get_array('data.adc', astype=np.float32)
    assert arr.dtype == np.float32

    assert jf['data.adc'].shape == (8, 2, 16, 512, 1024)
    assert jf['data.adc'].buffer_shape(
        module_gaps=True, roi=np.s_[:, :25, :35]
    ) == (8, 2, 16, 25, 35)


def test_jungfraus_first_modno(mock_jungfrau_run, mock_fxe_jungfrau_run):

    # Test SPB_IRDA_JF4M component by setting the first_modno to the default value 1.
    run = RunDirectory(mock_jungfrau_run)
    jf = JUNGFRAU(run.select_trains(by_index[:2]), first_modno=1)
    assert jf.detector_name == 'SPB_IRDA_JF4M'
    assert jf.n_modules == 8

    arr = jf.get_array('data.adc')
    assert np.all(arr['module'] == list(range(1, 9)))

    # Test FXE_XAD_JF500K component with and without setting first_modno to 3.
    for first_modno, modno in zip([1, 3], [3, 1]):
        run = RunDirectory(mock_fxe_jungfrau_run)
        jf = JUNGFRAU(
            run.select_trains(by_index[:2]),
            detector_name='FXE_XAD_JF500K',
            first_modno=first_modno,
        )
        assert jf.detector_name == 'FXE_XAD_JF500K'
        assert jf.n_modules == modno

        arr = jf.get_array('data.adc')
        assert np.all(arr['module'] == [modno])


def test_jungfrau_masked_data(mock_fxe_jungfrau_run):
    run = RunDirectory(mock_fxe_jungfrau_run)
    jf = JUNGFRAU(run, 'FXE_XAD_JF500K')

    # Default options
    kd = jf.masked_data().select_trains(np.s_[:1])
    arr = kd.ndarray()
    assert arr.shape == (1, 1, 16, 512, 1024)
    assert arr.dtype == np.float32
    line0 = np.zeros(1024, dtype=np.float32)
    line0[1:32] = np.nan
    np.testing.assert_array_equal(arr[0, 0, 0, 0, :], line0)

    # Xarray
    xarr = kd.xarray()
    assert xarr.dims[:2] == ('module', 'trainId')
    np.testing.assert_array_equal(xarr.values[0, 0, 0, 0, :], line0)

    # Specify which mask bits to use, & replace masked values with 99
    kd = jf.masked_data(mask_bits=1, masked_value=99).select_trains(np.s_[:1])
    arr = kd.ndarray()
    assert arr.shape == (1, 1, 16, 512, 1024)
    line0 = np.zeros(1024, dtype=np.float32)
    line0[1:32:2] = 99
    np.testing.assert_array_equal(arr[0, 0, 0, 0, :], line0)

    # Different field
    kd = jf.masked_data('data.gain', masked_value=255).select_trains(np.s_[:1])
    arr = kd.ndarray()
    assert arr.shape == (1, 1, 16, 512, 1024)
    assert arr.dtype == np.uint8
    line0 = np.zeros(1024, dtype=np.uint8)
    line0[1:32] = 255
    np.testing.assert_array_equal(arr[0, 0, 0, 0, :], line0)


def test_xtdf_masked_data(mock_reduced_spb_proc_run):
    run = RunDirectory(mock_reduced_spb_proc_run)
    agipd = AGIPD1M(run, modules=[8, 9])

    kd = agipd.masked_data().select_trains(np.s_[:1])
    assert kd.shape == (2, kd.shape[1], 512, 128)
    arr = kd.ndarray()
    assert arr.shape == kd.shape
    assert arr.dtype == np.float32
    line0_2mod = np.zeros((2, 128), dtype=np.float32)
    line0_2mod[1, 1:32] = np.nan
    np.testing.assert_array_equal(arr[:, 0, 0, :], line0_2mod)

    # Test with pulse selection (frames per train is consistent but arbitrary)
    kd_pulse_sel = kd.select_pulses(np.s_[:3])
    assert kd_pulse_sel.shape[1] <= 3
    assert kd_pulse_sel.ndarray().shape == kd_pulse_sel.shape

    kd = agipd.masked_data(mask_bits=[1, 4], masked_value=-1).select_trains(np.s_[:1])
    arr = kd.ndarray()
    line0_2mod = np.zeros((2, 128), dtype=np.float32)
    line0_2mod[1, np.nonzero(np.arange(32) & 5)] = -1
    np.testing.assert_array_equal(arr[:, 0, 0, :], line0_2mod)


def test_masked_data_raw_error(mock_fxe_raw_run):
    run = RunDirectory(mock_fxe_raw_run)
    lpd = LPD1M(run)

    with pytest.raises(RuntimeError, match="image.mask"):
        lpd.masked_data()


def test_get_dask_array(mock_fxe_raw_run):
    run = RunDirectory(mock_fxe_raw_run)
    det = LPD1M(run)
    arr = det.get_dask_array('image.data', fill_value=42)

    assert isinstance(arr.data, da.Array)
    assert arr.shape == (16, 480 * 128, 1, 256, 256)
    assert arr.dtype == np.uint16
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


def test_get_dask_array_lpd_parallelgain(mock_lpd_parallelgain_run):
    run = RunDirectory(mock_lpd_parallelgain_run)
    det = LPD1M(run.select_trains(by_index[:2]), parallel_gain=True)
    assert det.detector_name == 'FXE_DET_LPD1M-1'

    arr = det.get_dask_array('image.data')
    assert arr.shape == (16, 2 * 3 * 100, 1, 256, 256)
    assert arr.dims[:2] == ('module', 'train_pulse')
    np.testing.assert_array_equal(arr.coords['pulseId'], np.tile(np.arange(100), 6))


def test_get_dask_array_jungfrau(mock_jungfrau_run):
    run = RunDirectory(mock_jungfrau_run)
    jf = JUNGFRAU(run)
    assert jf.detector_name == 'SPB_IRDA_JF4M'

    arr = jf.get_dask_array('data.adc')
    assert arr.shape == (8, 100, 16, 512, 1024)
    assert arr.dims == ('module', 'train', 'cell', 'slow_scan', 'fast_scan')
    np.testing.assert_array_equal(arr.coords['train'], np.arange(10000, 10100))


def test_data_availability_lpd_gap(mock_lpd_mini_gap_run):
    run = RunDirectory(mock_lpd_mini_gap_run)
    det = LPD1M(run, modules=[0, 1])  # This example just contains 2 modules

    av = det.data_availability()
    assert av.shape == (2, 50)
    np.testing.assert_array_equal(av[1, 20:30], False)
    assert av.sum() == 2 * 50 - 10

    av_gaps = det.data_availability(module_gaps=True)
    assert av_gaps.shape == (16, 50)
    np.testing.assert_array_equal(av_gaps[2:], False)
    assert av_gaps.sum() == 2 * 50 - 10


def test_pulse_id_cell_id(mock_lpd_mini_gap_run):
    run = RunDirectory(mock_lpd_mini_gap_run)
    det = LPD1M(run, modules=[0, 1])  # This example just contains 2 modules
    kd = det['image.data']

    np.testing.assert_array_equal(
        kd.pulse_id_coordinates(), np.tile(np.arange(10), 5)
    )
    np.testing.assert_array_equal(
        kd.cell_id_coordinates(), np.tile(np.arange(10), 5)
    )

    np.testing.assert_array_equal(
        kd.select_pulses(np.s_[2:5]).pulse_id_coordinates(), np.tile(np.arange(2, 5), 5)
    )
    np.testing.assert_array_equal(
        kd.select_pulses(np.s_[2:5]).cell_id_coordinates(), np.tile(np.arange(2, 5), 5)
    )

def test_pulse_id_cell_id_reduced(mock_reduced_spb_proc_run):
    run = RunDirectory(mock_reduced_spb_proc_run)
    det = AGIPD1M(run)
    kd = det['image.data']
    nframes = kd.shape[1]

    # The selected frames are random, so we don't know precisely the pattern
    assert kd.train_id_coordinates().shape == (nframes,)
    assert kd.pulse_id_coordinates().shape == (nframes,)
    assert kd.cell_id_coordinates().shape == (nframes,)


def test_jungfrau_cell_ids(mock_jungfrau_run):
    run = RunDirectory(mock_jungfrau_run)
    det = JUNGFRAU(run)
    cellids = det.cell_ids()

    assert cellids.shape == (16,)


def test_select_trains(mock_fxe_raw_run):
    run = RunDirectory(mock_fxe_raw_run)
    det = LPD1M(run.select_trains(np.s_[:20]))
    assert len(det.train_ids) == 20
    det = det.select_trains(np.s_[:2])
    assert len(det.train_ids) == 2

    arr = det.get_array('image.data')
    assert arr.shape == (16, 2, 128, 256, 256)


def test_keydata_select_trains(mock_fxe_raw_run):
    run = RunDirectory(mock_fxe_raw_run)
    det = LPD1M(run.select_trains(np.s_[:20]))
    kd = det['image.data']
    assert len(kd.train_ids) == 20
    assert kd.shape == (16, 20 * 128, 256, 256)
    kd = kd[:3]
    assert len(kd.train_ids) == 3
    assert kd.shape == (16, 3 * 128, 256, 256)

    with pytest.raises(TypeError):
        iter(kd)


def test_split_trains(mock_fxe_raw_run):
    run = RunDirectory(mock_fxe_raw_run)
    det = LPD1M(run.select_trains(np.s_[:20]))
    assert len(det.train_ids) == 20

    parts = list(det.split_trains(parts=4))
    assert len(parts) == 4
    assert [len(p.train_ids) for p in parts] == [5, 5, 5, 5]
    assert sum([p.train_ids for p in parts], []) == det.train_ids

    arr = parts[-1].get_array('image.data', pulses=np.s_[:1])
    assert arr.shape == (16, 5, 1, 256, 256)

    # Split by a number of frames
    parts = list(det.split_trains(frames_per_part=256))
    assert [len(p.train_ids) for p in parts] == [2] * 10

    # frames_per_part less than one train (128 frames in this example)
    parts = list(det.split_trains(frames_per_part=100))
    assert [len(p.train_ids) for p in parts] == [1] * 20

    # trains_per_part cuts off before frames_per_part
    parts = list(det.split_trains(trains_per_part=3, frames_per_part=1024))
    assert [len(p.train_ids) for p in parts] == ([3] * 6) + [2]

    # parts cuts off before frames_per_part
    parts = list(det.split_trains(parts=6, frames_per_part=1024))
    assert [len(p.train_ids) for p in parts] == ([3] * 6) + [2]

    # frames_per_part > all frames in selection
    parts = list(det.split_trains(frames_per_part=3000))
    assert [len(p.train_ids) for p in parts] == [20]


def test_split_trains_jungfrau(mock_jungfrau_run):
    run = RunDirectory(mock_jungfrau_run)
    jf = JUNGFRAU(run.select_trains(np.s_[:20]))
    assert jf.frames_per_train == 16

    parts = list(jf.split_trains(frames_per_part=64))
    assert [len(p.train_ids) for p in parts] == [4] * 5


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


def test_iterate_lpd_parallel_gain(mock_lpd_parallelgain_run):
    run = RunDirectory(mock_lpd_parallelgain_run)
    det = LPD1M(run.select_trains(by_index[:3]), parallel_gain=True)

    tid, d = next(iter(det.trains()))
    assert d['image.data'].shape == (16, 1, 3, 100, 256, 256)
    assert d['image.data'].dims == \
           ('module', 'train', 'gain', 'pulse', 'slow_scan', 'fast_scan')


def test_iterate_jungfrau(mock_jungfrau_run):
    run = RunDirectory(mock_jungfrau_run)
    jf = JUNGFRAU(run)

    tid, d = next(iter(jf.trains()))
    assert tid == 10000
    assert d['data.adc'].shape == (8, 16, 512, 1024)
    assert d['data.adc'].dims == ('module', 'cell', 'slow_scan', 'fast_scan')


def test_modern_corr_sources(mock_modern_spb_proc_run, mock_spb_raw_run_fmt1):
    run_raw = RunDirectory(mock_spb_raw_run_fmt1)
    run_proc = RunDirectory(mock_modern_spb_proc_run)
    combined = run_raw.union(run_proc.select("*/CORR/*:output"))

    corr_sources = {f'SPB_DET_AGIPD1M-1/CORR/{i}CH0:output' for i in range(16)}
    det_sources = {f'SPB_DET_AGIPD1M-1/DET/{i}CH0:xtdf' for i in range(16)}

    # Specify that we want raw data
    assert AGIPD1M(run_raw, raw=True).data.all_sources == det_sources
    with pytest.raises(Exception):
        AGIPD1M(run_proc, raw=True)
    agipd_raw = AGIPD1M(combined, raw=True)
    assert agipd_raw.data.all_sources == det_sources
    assert 'image.mask' not in agipd_raw

    # Specify that we want corrected data
    with pytest.raises(Exception):
        AGIPD1M(run_raw, raw=False)
    assert AGIPD1M(run_proc, raw=False).data.all_sources == corr_sources
    agipd_proc = AGIPD1M(combined, raw=False)
    assert agipd_proc.data.all_sources == corr_sources
    assert 'image.mask' in agipd_proc

    # Legacy behaviour: prefer corrected, allow raw if only that is found
    assert AGIPD1M(run_raw).data.all_sources == det_sources
    assert AGIPD1M(run_proc).data.all_sources == corr_sources
    agipd_dflt = AGIPD1M(combined)
    assert agipd_dflt.data.all_sources == corr_sources
    assert 'image.mask' in agipd_dflt


def test_decompress_threads(mock_modern_spb_proc_run):
    run = RunDirectory(mock_modern_spb_proc_run)

    agipd = AGIPD1M(run[:1])
    # Load
    ref = agipd['image.mask'].ndarray(decompress_threads=1)
    print(ref.shape)
    print(ref[15, 0, :10, :10])
    assert ref[15, 0, 0, 0] == 0
    assert ref[15, 0, 0, 5] == 1

    import h5py
    h5py._errors.unsilence_errors()
    arr = agipd['image.mask'].ndarray(decompress_threads=16)
    np.testing.assert_array_equal(arr, ref)


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

def test_write_virtual_cxi_jungfrau(mock_jungfrau_run, tmpdir):
    run = RunDirectory(mock_jungfrau_run)
    det = JUNGFRAU(run)

    test_file = osp.join(str(tmpdir), 'test.cxi')
    det.write_virtual_cxi(test_file)
    assert_isfile(test_file)

    with h5py.File(test_file, 'r') as f:
        det_grp = f['entry_1/instrument_1/detector_1']
        ds = det_grp['data']
        assert isinstance(ds, h5py.Dataset)
        assert ds.is_virtual
        assert ds.shape[1:] == (8, 512, 1024)
        assert 'axes' in ds.attrs

        assert len(ds.virtual_sources()) == 8

        # Check position of each source file in the modules dimension
        for src in ds.virtual_sources():
            start, _, block, count = src.vspace.get_regular_hyperslab()
            assert block[1] == 1
            assert count[1] == 1

            expected_file = 'RAW-R0012-JNGFR{:0>2}-S00000.h5'.format(
                start[1] + 1)
            assert osp.basename(src.file_name) == expected_file

        # Check presence of other datasets
        assert 'gain' in det_grp
        assert 'mask' in det_grp
        assert 'experiment_identifier' in det_grp

def test_write_virtual_cxi_jungfrau_some_modules(mock_jungfrau_run, tmpdir):
    run = RunDirectory(mock_jungfrau_run)
    det = JUNGFRAU(run, modules=[2, 3, 4, 6])

    test_file = osp.join(str(tmpdir), 'test.cxi')
    det.write_virtual_cxi(test_file)
    assert_isfile(test_file)

    with h5py.File(test_file, 'r') as f:
        det_grp = f['entry_1/instrument_1/detector_1']
        ds = det_grp['data']
        assert ds.shape[1:] == (8, 512, 1024)
        np.testing.assert_array_equal(det_grp['module_identifier'][:], np.arange(1,9))

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

    trains = np.repeat(np.arange(10000, 10006), 2)
    pulses = np.tile([0, 5], 6)
    test_file = tmp_path / 'sel_frames.h5'
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

    # pytest leaves temp files for inspection, but these files are big enough
    # to be inconvenient, so delete them if the assertions have passed.
    test_file.unlink()


def test_write_selected_frames_proc(mock_spb_proc_run, tmp_path):
    run = RunDirectory(mock_spb_proc_run)
    det = AGIPD1M(run)

    trains = np.repeat(np.arange(10000, 10006), 2)
    pulses = np.tile([0, 7], 6)
    test_file = tmp_path / 'sel_frames.h5'
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

    # pytest leaves temp files for inspection, but these files are big enough
    # to be inconvenient, so delete them if the assertions have passed.
    test_file.unlink()

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
