import os
import numpy as np
import xarray as xr
import pytest

import h5py

from extra_data import RunDirectory, H5File
from extra_data.keydata import expand_indexing
from extra_data.exceptions import TrainIDError, NoDataError
from . import make_examples
from .mockdata import write_file
from .mockdata.xgm import XGM


def test_get_keydata(mock_spb_raw_run):
    run = RunDirectory(mock_spb_raw_run)
    print(run.instrument_sources)
    am0 = run['SPB_DET_AGIPD1M-1/DET/0CH0:xtdf', 'image.data']
    assert len(am0.files) == 1
    assert am0.section == 'INSTRUMENT'
    assert am0.is_instrument
    assert am0.entry_shape == (2, 512, 128)
    assert am0.ndim == 4
    assert am0.dtype == np.dtype('u2')
    assert {p.name for p in am0.source_file_paths} == {
        'RAW-R0238-AGIPD00-S00000.h5'
    }

    xgm_beam_x = run['SPB_XTD9_XGM/DOOCS/MAIN', 'beamPosition.ixPos.value']
    assert len(xgm_beam_x.files) == 2
    assert xgm_beam_x.section == 'CONTROL'
    assert xgm_beam_x.is_control
    assert xgm_beam_x.entry_shape == ()
    assert xgm_beam_x.ndim == 1
    assert xgm_beam_x.dtype == np.dtype('f4')
    assert {p.name for p in xgm_beam_x.source_file_paths} == {
        'RAW-R0238-DA01-S00000.h5', 'RAW-R0238-DA01-S00001.h5'
    }

    data = xgm_beam_x.ndarray()
    assert xgm_beam_x.nbytes == data.nbytes

    # Ensure KeyData is not accidentally iterable
    with pytest.raises(TypeError):
        iter(xgm_beam_x)


def test_select_trains(mock_spb_raw_run):
    run = RunDirectory(mock_spb_raw_run)
    xgm_beam_x = run['SPB_XTD9_XGM/DOOCS/MAIN', 'beamPosition.ixPos.value']
    assert xgm_beam_x.shape == (64,)

    sel1 = xgm_beam_x[:20]  # Equivalent to .select_trains(np.s_[:20])
    assert sel1.shape == (20,)
    assert len(sel1.files) == 1

    # Empty selection
    sel2 = xgm_beam_x[80:]
    assert sel2.shape == (0,)
    assert len(sel2.files) == 1
    assert sel2.xarray().shape == (0,)

    # Single train
    sel3 = xgm_beam_x[32]
    assert sel3.shape == (1,)


def test_split_trains(mock_spb_raw_run):
    run = RunDirectory(mock_spb_raw_run)
    xgm_beam_x = run['SPB_XTD9_XGM/DOOCS/MAIN', 'beamPosition.ixPos.value']
    assert xgm_beam_x.shape == (64,)

    chunks = list(xgm_beam_x.split_trains(3))
    assert len(chunks) == 3
    assert {c.shape for c in chunks} == {(21,), (22,)}
    assert chunks[0].ndarray().shape == chunks[0].shape

    chunks = list(xgm_beam_x.split_trains(3, trains_per_part=20))
    assert len(chunks) == 4
    assert {c.shape for c in chunks} == {(16,)}


def test_nodata(mock_fxe_raw_run):
    run = RunDirectory(mock_fxe_raw_run)
    cam_pix = run['FXE_XAD_GEC/CAM/CAMERA_NODATA:daqOutput', 'data.image.pixels']

    assert cam_pix.train_ids == list(range(10000, 10480))
    assert len(cam_pix.files) == 2
    assert cam_pix.shape == (0, 255, 1024)

    arr = cam_pix.xarray()
    assert arr.shape == (0, 255, 1024)
    assert arr.dtype == np.dtype('u2')

    dask_arr = cam_pix.dask_array(labelled=True)
    assert dask_arr.shape == (0, 255, 1024)
    assert dask_arr.dtype == np.dtype('u2')

    assert list(cam_pix.trains()) == []
    tid, data = cam_pix.train_from_id(10010)
    assert tid == 10010
    assert data.shape == (0, 255, 1024)


def test_iter_trains(mock_spb_raw_run):
    run = RunDirectory(mock_spb_raw_run)
    xgm_beam_x = run['SPB_XTD9_XGM/DOOCS/MAIN', 'beamPosition.ixPos.value']

    assert [t for (t, _) in xgm_beam_x.trains()] == list(range(10000, 10064))
    for _, v in xgm_beam_x.trains():
        assert isinstance(v, np.float32)
        break


def test_iter_trains_keep_dims(mock_jungfrau_run):
    run = RunDirectory(mock_jungfrau_run)
    jf_data = run['SPB_IRDA_JF4M/DET/JNGFR01:daqOutput', 'data.adc']

    for _, v in jf_data.trains(keep_dims=True):
        assert v.shape == (1, 16, 512, 1024)


def test_iter_trains_include_empty(mock_sa3_control_data):
    f = H5File(mock_sa3_control_data)
    beamview = f['SA3_XTD10_IMGFEL/CAM/BEAMVIEW2:daqOutput', 'data.image.dims']

    for expected_tid, (data1_tid, data1), (data2_tid, data2) in zip(
        beamview.train_ids,
        beamview.trains(include_empty=True),
        beamview.trains(include_empty=True, keep_dims=True)
    ):
        assert expected_tid == data1_tid == data2_tid

        if (expected_tid % 2) == 0:
            assert data1 is None
        else:
            assert data1.shape == (2,)

        assert data2.shape == (expected_tid % 2, 2)


def test_get_train(mock_spb_raw_run):
    run = RunDirectory(mock_spb_raw_run)
    xgm_beam_x = run['SPB_XTD9_XGM/DOOCS/MAIN', 'beamPosition.ixPos.value']

    tid, val = xgm_beam_x.train_from_id(10005)
    assert tid == 10005
    assert isinstance(val, np.float32)

    with pytest.raises(TrainIDError):
        xgm_beam_x.train_from_id(11000)

    tid, _ = xgm_beam_x.train_from_index(-10)
    assert tid == 10054

    with pytest.raises(IndexError):
        xgm_beam_x.train_from_index(9999)


def test_get_train_keep_dims(mock_jungfrau_run):
    run = RunDirectory(mock_jungfrau_run)
    jf_adc = run['SPB_IRDA_JF4M/DET/JNGFR01:daqOutput', 'data.adc']

    _, val = jf_adc.train_from_id(10005, keep_dims=True)
    assert val.shape == (1, 16, 512, 1024)

    _, val = jf_adc.train_from_index(-10, keep_dims=True)
    assert val.shape == (1, 16, 512, 1024)


def test_data_counts(mock_reduced_spb_proc_run):
    run = RunDirectory(mock_reduced_spb_proc_run)

    # control data
    xgm_beam_x = run['SPB_XTD9_XGM/DOOCS/MAIN', 'beamPosition.ixPos.value']
    count = xgm_beam_x.data_counts()
    assert count.index.tolist() == xgm_beam_x.train_ids
    assert (count.values == 1).all()

    # instrument data
    camera = run['SPB_IRU_CAM/CAM/SIDEMIC:daqOutput', 'data.image.pixels']
    count = camera.data_counts()
    assert count.index.tolist() == camera.train_ids

    mod = run['SPB_DET_AGIPD1M-1/DET/0CH0:xtdf', 'image.data']
    count = mod.data_counts()
    assert count.index.tolist() == mod.train_ids
    assert count.values.sum() == mod.shape[0]


def test_data_counts_empty(mock_fxe_raw_run):
    run = RunDirectory(mock_fxe_raw_run)
    cam_nodata = run['FXE_XAD_GEC/CAM/CAMERA_NODATA:daqOutput', 'data.image.pixels']

    count_ser = cam_nodata.data_counts(labelled=True)
    assert len(count_ser) == 480
    assert count_ser.sum() == 0

    count_arr = cam_nodata.data_counts(labelled=False)
    assert len(count_arr) == 480
    assert count_arr.sum() == 0

    count_none_ser = cam_nodata.drop_empty_trains().data_counts(labelled=True)
    assert len(count_none_ser) == 0

    count_none_arr = cam_nodata.drop_empty_trains().data_counts(labelled=False)
    assert len(count_none_arr) == 0


@pytest.fixture()
def fxe_run_module_offset(tmp_path):
    run_dir = tmp_path / 'fxe-run-mod-offset'
    run_dir.mkdir()
    make_examples.make_fxe_run(run_dir, format_version='1.0')

    # Shift the train IDs for a module by 1, so it has data for a different set
    # of train IDs to other sources.
    with h5py.File(run_dir / 'RAW-R0450-LPD08-S00000.h5', 'r+') as f:
        tids_dset = f['INDEX/trainId']
        tids_dset[:] = tids_dset[:] + 1

    return run_dir


def test_data_counts_missing_train(fxe_run_module_offset):
    run = RunDirectory(fxe_run_module_offset)
    assert len(run.train_ids) == 481
    lpd_m8 = run['FXE_DET_LPD1M-1/DET/8CH0:xtdf', 'image.cellId']

    ser = lpd_m8.data_counts(labelled=True)
    assert len(ser) == 480
    np.testing.assert_array_equal(ser.index, run.train_ids[1:])

    arr = lpd_m8.data_counts(labelled=False)
    assert len(arr) == 481
    assert arr[0] == 0
    np.testing.assert_array_equal(arr[1:], 128)

    lpd_m8_w_data = lpd_m8.drop_empty_trains()
    ser = lpd_m8_w_data.data_counts(labelled=True)
    assert len(ser) == 480
    np.testing.assert_array_equal(ser.index, run.train_ids[1:])

    arr = lpd_m8_w_data.data_counts(labelled=False)
    assert len(arr) == 480
    np.testing.assert_array_equal(arr, 128)


def test_select_by(mock_spb_raw_run):
    run = RunDirectory(mock_spb_raw_run)
    am0 = run['SPB_DET_AGIPD1M-1/DET/0CH0:xtdf', 'image.data']

    subrun = run.select(am0)
    assert subrun.all_sources == {am0.source}
    assert subrun.keys_for_source(am0.source) == {am0.key}


def test_drop_empty_trains(mock_sa3_control_data):
    f = H5File(mock_sa3_control_data)
    beamview = f['SA3_XTD10_IMGFEL/CAM/BEAMVIEW2:daqOutput', 'data.image.dims']

    assert len(beamview.train_ids) == 500
    a1 = beamview.ndarray()
    assert a1.shape == (250, 2)
    frame_counts = beamview.data_counts(labelled=False)
    assert frame_counts.shape == (500,)
    assert frame_counts.min() == 0

    beamview_w_data = beamview.drop_empty_trains()
    assert len(beamview_w_data.train_ids) == 250
    np.testing.assert_array_equal(beamview_w_data.ndarray(), a1)
    frame_counts = beamview_w_data.data_counts(labelled=False)
    assert frame_counts.shape == (250,)
    assert frame_counts.min() == 1


def test_single_value(mock_sa3_control_data, monkeypatch):
    f = H5File(mock_sa3_control_data)

    imager = f['SA3_XTD10_IMGFEL/CAM/BEAMVIEW:daqOutput', 'data.image.pixels']
    flux = f['SA3_XTD10_XGM/XGM/DOOCS', 'pulseEnergy.photonFlux']
    state = f['SA3_XTD10_XGM/XGM/DOOCS', 'state']

    # Try without data for a source and key.
    with pytest.raises(NoDataError):
        imager.as_single_value()  # FEL imager with no data.

    with pytest.raises(NoDataError):
        flux[:0].as_single_value()  # No data through selection.

    # Monkeypatch some actual data into the KeyData object
    data = np.arange(flux.shape[0])
    monkeypatch.setattr(flux, 'ndarray', lambda: data)

    # Try some tolerances that have to fail.
    with pytest.raises(ValueError):
        flux.as_single_value()

    with pytest.raises(ValueError):
        flux.as_single_value(atol=1)

    with pytest.raises(ValueError):
        flux.as_single_value(rtol=0.1)

    # Try with large enough tolerances.
    assert flux.as_single_value(atol=len(data)/2) == np.median(data)
    assert flux.as_single_value(rtol=0.5, atol=len(data)/4) == np.median(data)
    assert flux.as_single_value(rtol=1) == np.median(data)

    # Other reduction options
    assert flux.as_single_value(rtol=1, reduce_by='mean') == np.mean(data)
    assert flux.as_single_value(rtol=1, reduce_by=np.mean) == np.mean(data)
    assert flux.as_single_value(atol=len(data)-1, reduce_by='first') == 0

    # Try strings.
    assert state[5:].as_single_value() == 'ON'

    with pytest.raises(ValueError):
        # Contains two unique values.
        state.as_single_value()

    with pytest.raises(TypeError):
        # Does not accept reduce_by
        state.as_single_value(reduce_by='mean')

    # Try vector data.
    intensity = f['SA3_XTD10_XGM/XGM/DOOCS:output', 'data.intensityTD']
    data = np.repeat(data, intensity.shape[1]).reshape(-1, intensity.shape[-1])
    monkeypatch.setattr(intensity, 'ndarray', lambda: data)

    with pytest.raises(ValueError):
        intensity.as_single_value()

    np.testing.assert_equal(intensity.as_single_value(rtol=1), np.median(data))


def test_ndarray_out(mock_spb_raw_run):
    f = RunDirectory(mock_spb_raw_run)
    cam = f['SPB_IRU_CAM/CAM/SIDEMIC:daqOutput', 'data.image.dims']

    buf_new = cam.ndarray()  # New copy of data.
    buf_in = np.zeros(cam.shape, dtype=cam.dtype)
    buf_out = cam.ndarray(out=buf_in)  # In-place copy of data.

    np.testing.assert_allclose(buf_new, buf_out)
    assert buf_in is buf_out


def test_string_arrays(mock_spb_raw_run):
    f = RunDirectory(mock_spb_raw_run)
    state = f['SPB_XTD9_XGM/DOOCS/MAIN', 'state']

    for data in [state.ndarray(), state.xarray(), state.series()]:
        assert data.dtype.hasobject
        assert (data[3:8] == ['OFF', 'OFF', 'ON', 'ON', 'ON']).all()


def test_xarray_structured_data(mock_remi_run):
    run = RunDirectory(mock_remi_run)
    dset = run['SQS_REMI_DLD6/DET/TOP:output', 'rec.hits'].xarray()

    assert isinstance(dset, xr.Dataset)
    assert list(dset.data_vars.keys()) == ['x', 'y', 't', 'm']

    arrs = list(dset.data_vars.values())

    assert all([arr.shape == (100, 50) for arr in arrs])
    assert all([arr.dtype == np.float64 for arr in arrs[:3]])
    assert arrs[3].dtype == np.int32

    np.testing.assert_equal(dset.coords['trainId'], np.arange(10000, 10100))

# @pytest.mark.parametrize("extra_dims", [None, ("a", "b", "c")], ids=["default dims", "custom dims"])
# def test_xarray_coords(mock_spb_raw_run, extra_dims):
#     run = RunDirectory(mock_spb_raw_run)
#     am0 = run['SPB_DET_AGIPD1M-1/DET/0CH0:xtdf', 'image.data'][0]
#     assert am0.entry_shape == (2, 512, 128)

#     def validate_coords(data_array, coords=None):
#         coords = coords or {}
#         coords_names = list(coords)
#         dims = ['trainId']
#         if extra_dims is not None:
#             dims += extra_dims
#         else:
#             dims += [f'dim_{idx}' for idx in range()

#         if extra_dims is None:
#             assert list(data_array.coords) == ['trainId']
#         else:
#             assert data_array.coords == {'trainId'}.union(extra_dims)

#     # ROI with slices, extra_dims -> generate coords
#     darr1 = am0.xarray(roi=np.s_[:, 10:20, 5:15], extra_dims=extra_dims)
#     assert darr1.shape == (64, 2, 10, 10)
#     validate_coords(darr1)
#     np.testing.assert_array_equal(darr1.coords['b'], np.arange(10, 20))
#     np.testing.assert_array_equal(darr1.coords['c'], np.arange(5, 15))

#     # ROI with an integer to drop a dimension, no extra_dims argument -> no extra coords
#     darr2 = am0.xarray(roi=np.s_[:, 10, 5:15])
#     assert darr2.shape == (64, 2, 10)
#     assert darr2.coords == {'trainId'}
#     np.testing.assert_array_equal(darr2.coords['dim_2'], np.arange(5, 15))

#     # Short ROI: explicit non-full slice
#     darr3 = am0.xarray(roi=np.s_[:1, 10:20], extra_dims=['a', 'b', 'c'])
#     assert darr3.shape == (64, 1, 10, 128)
#     np.testing.assert_array_equal(darr3.coords['a'], np.arange(0, 1))
#     np.testing.assert_array_equal(darr3.coords['b'], np.arange(10, 20))

#     # No ROI defined
#     darr4 = am0.xarray(extra_dims=['a', 'b', 'c'])
#     assert darr4.shape == (64, 2, 512, 128)
#     assert set(darr4.coords) == {'trainId', 'a', 'b', 'c'}

#     # Ellipsis
#     darr5 = am0.xarray(roi=np.s_[:1, 30:40, ...], extra_dims=['a', 'b', 'c'])
#     assert darr5.shape == (64, 1, 10, 128)
#     np.testing.assert_array_equal(darr5.coords['a'], np.arange(0, 1))
#     np.testing.assert_array_equal(darr5.coords['b'], np.arange(30, 40))
#     np.testing.assert_array_equal(darr5.coords['c'], np.arange(128))

#     darr6 = am0.xarray(roi=np.s_[..., 30:40], extra_dims=['a', 'b', 'c'])
#     assert darr6.shape == (64, 2, 512, 10)
#     np.testing.assert_array_equal(darr6.coords['a'], np.arange(2))
#     np.testing.assert_array_equal(darr6.coords['b'], np.arange(512))
#     np.testing.assert_array_equal(darr6.coords['c'], np.arange(30, 40))

#     # Integer-only ROI: drops first entry dim
#     darr7 = am0.xarray(roi=np.s_[0])
#     assert darr7.shape == (64, 512, 128)
#     assert set(darr7.coords) == {'trainId'}
#     assert darr7.dims == ('trainId', 'dim_1', 'dim_2')

#     # list of indices
#     darr8 = am0.xarray(roi=np.s_[0, [30, 33], :])
#     assert darr8.shape == (64, 2, 128)
#     # The first remaining dim corresponds to advanced indexer
#     np.testing.assert_array_equal(darr8.coords['dim_1'], np.array([30, 33]))
#     assert darr8.coords == {'trainId'}
#     assert darr8.dims == ('trainId', 'dim_1', 'dim_2')

#     # boolean indexing
#     mask = np.zeros((512,), dtype=np.bool_)
#     mask[30:40] = True
#     darr9 = am0.xarray(roi=np.s_[0, mask])
#     assert darr9.shape == (64, 10, 128)
#     np.testing.assert_array_equal(darr9.coords['dim_1'], np.arange(30, 40))
#     assert darr9.dims == ('trainId', 'dim_1', 'dim_2')

#     darr9 = am0.xarray(roi=np.s_[[False, True], 1, 2])
#     assert darr9.shape == (64, 1)
#     np.testing.assert_array_equal(darr9.coords['dim_0'], np.arange(1, 2))
#     assert darr9.dims == ('trainId', 'dim_0')

#     mask = np.zeros((512,), dtype=np.bool_)
#     mask[::2] = True
#     darr10 = am0.xarray(roi=np.s_[1, mask, 5:15], extra_dims=['raw_proc', 'ss', 'fs'])
#     assert darr10.shape == (64, 256, 10)
#     np.testing.assert_array_equal(darr10.coords['ss'], np.arange(0, 512, 2))
#     np.testing.assert_array_equal(darr10.coords['fs'], np.arange(5, 15))
#     assert darr10.dims == ('trainId', 'ss', 'fs')

#     # negative indexing
#     darr11 = am0.xarray(roi=np.s_[:, -10:, [-10, -5, -1]], extra_dims=['a', 'b', 'c'])
#     assert darr11.shape == (64, 2, 10, 3)
#     np.testing.assert_array_equal(darr11.coords['b'], np.arange(502, 512))
#     np.testing.assert_array_equal(darr11.coords['c'], np.array([118, 123, 127]))
#     assert darr11.dims == ('trainId', 'a', 'b', 'c')


# def test_xarray_coords(mock_spb_raw_run):
#     run = RunDirectory(mock_spb_raw_run)
#     am0 = run['SPB_DET_AGIPD1M-1/DET/0CH0:xtdf', 'image.data'][0]
#     assert am0.entry_shape == (2, 512, 128)

#     # ROI with slices, extra_dims -> generate coords
#     darr1 = am0.xarray(roi=np.s_[:, 10:20, 5:15], extra_dims=['a', 'b', 'c'])
#     assert darr1.shape == (64, 2, 10, 10)
#     assert darr1.coords == {'trainId', 'a', 'b', 'c'}
#     np.testing.assert_array_equal(darr1.coords['b'], np.arange(10, 20))
#     np.testing.assert_array_equal(darr1.coords['c'], np.arange(5, 15))

#     # ROI with an integer to drop a dimension, no extra_dims argument -> no extra coords
#     darr2 = am0.xarray(roi=np.s_[:, 10, 5:15])
#     assert darr2.shape == (64, 2, 10)
#     assert darr2.coords == {'trainId'}
#     np.testing.assert_array_equal(darr2.coords['dim_2'], np.arange(5, 15))

#     # Short ROI: explicit non-full slice
#     darr3 = am0.xarray(roi=np.s_[:1, 10:20], extra_dims=['a', 'b', 'c'])
#     assert darr3.shape == (64, 1, 10, 128)
#     np.testing.assert_array_equal(darr3.coords['a'], np.arange(0, 1))
#     np.testing.assert_array_equal(darr3.coords['b'], np.arange(10, 20))

#     # No ROI defined
#     darr4 = am0.xarray(extra_dims=['a', 'b', 'c'])
#     assert darr4.shape == (64, 2, 512, 128)
#     assert set(darr4.coords) == {'trainId', 'a', 'b', 'c'}

#     # Ellipsis
#     darr5 = am0.xarray(roi=np.s_[:1, 30:40, ...], extra_dims=['a', 'b', 'c'])
#     assert darr5.shape == (64, 1, 10, 128)
#     np.testing.assert_array_equal(darr5.coords['a'], np.arange(0, 1))
#     np.testing.assert_array_equal(darr5.coords['b'], np.arange(30, 40))
#     np.testing.assert_array_equal(darr5.coords['c'], np.arange(128))

#     darr6 = am0.xarray(roi=np.s_[..., 30:40], extra_dims=['a', 'b', 'c'])
#     assert darr6.shape == (64, 2, 512, 10)
#     np.testing.assert_array_equal(darr6.coords['a'], np.arange(2))
#     np.testing.assert_array_equal(darr6.coords['b'], np.arange(512))
#     np.testing.assert_array_equal(darr6.coords['c'], np.arange(30, 40))

#     # Integer-only ROI: drops first entry dim
#     darr7 = am0.xarray(roi=np.s_[0])
#     assert darr7.shape == (64, 512, 128)
#     assert set(darr7.coords) == {'trainId'}
#     assert darr7.dims == ('trainId', 'dim_1', 'dim_2')

#     # list of indices
#     darr8 = am0.xarray(roi=np.s_[0, [30, 33], :])
#     assert darr8.shape == (64, 2, 128)
#     # The first remaining dim corresponds to advanced indexer
#     np.testing.assert_array_equal(darr8.coords['dim_1'], np.array([30, 33]))
#     assert darr8.coords == {'trainId'}
#     assert darr8.dims == ('trainId', 'dim_1', 'dim_2')

#     # boolean indexing
#     mask = np.zeros((512,), dtype=np.bool_)
#     mask[30:40] = True
#     darr9 = am0.xarray(roi=np.s_[0, mask])
#     assert darr9.shape == (64, 10, 128)
#     np.testing.assert_array_equal(darr9.coords['dim_1'], np.arange(30, 40))
#     assert darr9.dims == ('trainId', 'dim_1', 'dim_2')

#     darr9 = am0.xarray(roi=np.s_[[False, True], 1, 2])
#     assert darr9.shape == (64, 1)
#     np.testing.assert_array_equal(darr9.coords['dim_0'], np.arange(1, 2))
#     assert darr9.dims == ('trainId', 'dim_0')

#     mask = np.zeros((512,), dtype=np.bool_)
#     mask[::2] = True
#     darr10 = am0.xarray(roi=np.s_[1, mask, 5:15], extra_dims=['raw_proc', 'ss', 'fs'])
#     assert darr10.shape == (64, 256, 10)
#     np.testing.assert_array_equal(darr10.coords['ss'], np.arange(0, 512, 2))
#     np.testing.assert_array_equal(darr10.coords['fs'], np.arange(5, 15))
#     assert darr10.dims == ('trainId', 'ss', 'fs')

#     # negative indexing
#     darr11 = am0.xarray(roi=np.s_[:, -10:, [-10, -5, -1]], extra_dims=['a', 'b', 'c'])
#     assert darr11.shape == (64, 2, 10, 3)
#     np.testing.assert_array_equal(darr11.coords['b'], np.arange(502, 512))
#     np.testing.assert_array_equal(darr11.coords['c'], np.array([118, 123, 127]))
#     assert darr11.dims == ('trainId', 'a', 'b', 'c')

@pytest.fixture()
def run_with_file_no_trains(mock_spb_raw_run):
    extra_file = os.path.join(mock_spb_raw_run, 'RAW-R0238-DA01-S00002.h5')
    write_file(extra_file, [
        XGM('SPB_XTD9_XGM/DOOCS/MAIN'),
    ], ntrains=0)
    try:
        yield mock_spb_raw_run
    finally:
        os.unlink(extra_file)


def test_file_no_trains(run_with_file_no_trains):
    run = RunDirectory(run_with_file_no_trains)
    xpos = run['SPB_XTD9_XGM/DOOCS/MAIN', 'beamPosition.ixPos'].ndarray()
    assert xpos.shape == (64,)


def test_attributes(mock_sa3_control_data):
    run = H5File(mock_sa3_control_data)

    # INSTRUMENT key.
    xgm_intensity = run['SA3_XTD10_XGM/XGM/DOOCS:output', 'data.intensityTD']
    attrs = xgm_intensity.attributes()

    assert isinstance(attrs, dict)
    assert attrs['metricPrefixName'] == 'micro'
    assert attrs['unitSymbol'] == 'J'

    # CONTROL key.
    xgm_beampos_x = run['SA3_XTD10_XGM/XGM/DOOCS', 'beamPosition.ixPos']
    attrs = xgm_beampos_x.attributes()

    assert isinstance(attrs, dict)
    assert attrs['alias'] == 'IX.POS'
    assert attrs['description'] == 'Calculated X position [mm]'
    assert attrs['daqPolicy'][0] == -1


def test_units(mock_sa3_control_data):
    run = H5File(mock_sa3_control_data)
    xgm_intensity = run['SA3_XTD10_XGM/XGM/DOOCS:output', 'data.intensityTD']

    assert xgm_intensity.units == 'μJ'
    assert xgm_intensity.units_name == 'microjoule'

    # Check that it still works after selecting 0 trains
    assert xgm_intensity.select_trains(np.s_[:0]).units == 'μJ'

    # units are added to xarray's attributes
    assert xgm_intensity.xarray().attrs['units'] == 'μJ'


def test_expand_indexing():
    shape = (2, 512, 128)

    # Basic slices (including full slice)
    out = expand_indexing(shape, (slice(None), slice(10, 20), slice(5, 15)))
    np.testing.assert_array_equal(out[0], np.arange(2))
    np.testing.assert_array_equal(out[1], np.arange(10, 20))
    np.testing.assert_array_equal(out[2], np.arange(5, 15))

    # Ellipsis expansion and padding with full slice
    out = expand_indexing(shape, (slice(0, 1), slice(10, 12), Ellipsis))
    np.testing.assert_array_equal(out[0], np.arange(0, 1))
    np.testing.assert_array_equal(out[1], np.arange(10, 12))
    np.testing.assert_array_equal(out[2], np.arange(128))

    # Integer indexing and negative indices
    out = expand_indexing(shape, (1, slice(-10, None), [-10, -5, -1]))
    assert out[0] == 1
    np.testing.assert_array_equal(out[1], np.arange(502, 512))
    np.testing.assert_array_equal(out[2], np.array([118, 123, 127]))

    # Fancy integer array and boolean list
    idx_list = [0, 3, 7]
    out = expand_indexing(shape, (0, idx_list, slice(None)))
    assert out[0] == 0
    np.testing.assert_array_equal(out[1], np.array(idx_list))
    np.testing.assert_array_equal(out[2], np.arange(128))

    mask = np.zeros((512,), dtype=bool)
    mask[30:40] = True
    out = expand_indexing(shape, (0, mask, slice(5, 15)))
    assert out[0] == 0
    np.testing.assert_array_equal(out[1], np.arange(30, 40))
    np.testing.assert_array_equal(out[2], np.arange(5, 15))


def test_xarray_extra_dims_and_coords(mock_spb_raw_run):
    run = RunDirectory(mock_spb_raw_run)
    am0 = run['SPB_DET_AGIPD1M-1/DET/0CH0:xtdf', 'image.data'][0]
    assert am0.entry_shape == (2, 512, 128)

    # No extra dims -> only trainId coordinate
    da_default = am0.xarray()
    assert da_default.shape == (64, 2, 512, 128)
    assert da_default.dims == ('trainId', 'dim_0', 'dim_1', 'dim_2')
    assert set(da_default.coords) == {'trainId'}

    # ROI with slices, with extra_dims -> generate default coords
    da1 = am0.xarray(roi=np.s_[:, 10:20, 5:15], extra_dims=['raw_proc', 'ss', 'fs'])
    assert da1.shape == (64, 2, 10, 10)
    assert da1.dims == ('trainId', 'raw_proc', 'ss', 'fs')
    np.testing.assert_array_equal(da1.coords['raw_proc'], np.arange(2))
    np.testing.assert_array_equal(da1.coords['ss'], np.arange(10, 20))
    np.testing.assert_array_equal(da1.coords['fs'], np.arange(5, 15))

    # Short ROI: explicit non-full slice on first entry dim
    da2 = am0.xarray(roi=np.s_[:1, 30:40, ...], extra_dims=['a', 'b', 'c'])
    assert da2.shape == (64, 1, 10, 128)
    assert da2.dims == ('trainId', 'a', 'b', 'c')
    np.testing.assert_array_equal(da2.coords['a'], np.arange(0, 1))
    np.testing.assert_array_equal(da2.coords['b'], np.arange(30, 40))
    np.testing.assert_array_equal(da2.coords['c'], np.arange(128))

    # Ellipsis at the start
    da3 = am0.xarray(roi=np.s_[..., 30:40], extra_dims=['a', 'b', 'c'])
    assert da3.shape == (64, 2, 512, 10)
    assert da3.dims == ('trainId', 'a', 'b', 'c')
    np.testing.assert_array_equal(da3.coords['a'], np.arange(2))
    np.testing.assert_array_equal(da3.coords['b'], np.arange(512))
    np.testing.assert_array_equal(da3.coords['c'], np.arange(30, 40))

    # Integer-only index on first entry dim with extra_dims -> keep dim with size 1
    da4 = am0.xarray(roi=np.s_[0, :, :], extra_dims=['a', 'b', 'c'])
    # The data is a single index in the first entry, dim is dropped
    assert da4.shape == (64, 512, 128)
    assert da4.dims == ('trainId', 'b', 'c')
    # Coordinate for the dropped dimension is the selected index
    a_coord = np.asarray(da4.coords['a'])
    np.testing.assert_array_equal(a_coord, np.array(0))

    # Fancy indexing with list of indices
    da5 = am0.xarray(roi=np.s_[:, [30, 33], :], extra_dims=['a', 'b', 'c'])
    assert da5.shape == (64, 2, 2, 128)
    np.testing.assert_array_equal(da5.coords['b'], np.array([30, 33]))

    # Boolean indexing
    mask = np.zeros((512,), dtype=bool)
    mask[::2] = True
    da6 = am0.xarray(roi=np.s_[:, mask, 5:15], extra_dims=['a', 'ss', 'fs'])
    assert da6.shape == (64, 2, 256, 10)
    np.testing.assert_array_equal(da6.coords['ss'], np.arange(0, 512, 2))
    np.testing.assert_array_equal(da6.coords['fs'], np.arange(5, 15))

    # Negative indexing on last two dims
    da7 = am0.xarray(roi=np.s_[:, -10:, [-10, -5, -1]], extra_dims=['a', 'b', 'c'])
    assert da7.shape == (64, 2, 10, 3)
    np.testing.assert_array_equal(da7.coords['b'], np.arange(502, 512))
    np.testing.assert_array_equal(da7.coords['c'], np.array([118, 123, 127]))

    # Custom coordinates via extra_coords
    custom_fs = np.linspace(0.0, 9.0, 10)
    da8 = am0.xarray(roi=np.s_[:, 10:20, 5:15], extra_dims=['a', 'b', 'fs'], extra_coords={'fs': custom_fs})
    assert da8.shape == (64, 2, 10, 10)
    assert da8.dims == ('trainId', 'a', 'b', 'fs')
    assert 'a' not in da8.coords
    assert 'b' not in da8.coords
    np.testing.assert_allclose(da8.coords['fs'], custom_fs)

    # Passing only extra_coords (no extra_dims)
    with pytest.raises(ValueError):
        am0.xarray(roi=np.s_[:, 10:20, 5:15], extra_coords={'fs': np.arange(5, 15)})

    da9 = am0.xarray(roi=np.s_[:, 10:20, 5:15], extra_dims=['r/p', 'ss', 'fs'], extra_coords={'fs': np.arange(5, 15)})
    assert list(da9.coords) >= ['trainId', 'fs']

    da10 = am0.xarray(roi=np.s_[:, 10:20, 5:15], extra_coords={'dim_0': [100, 200]})
    assert list(da9.coords) >= ['trainId', 'dim_0']
