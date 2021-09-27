import numpy as np
import pytest

from extra_data import RunDirectory, H5File
from extra_data.exceptions import TrainIDError

def test_get_keydata(mock_spb_raw_run):
    run = RunDirectory(mock_spb_raw_run)
    print(run.instrument_sources)
    am0 = run['SPB_DET_AGIPD1M-1/DET/0CH0:xtdf', 'image.data']
    assert len(am0.files) == 1
    assert am0.section == 'INSTRUMENT'
    assert am0.entry_shape == (2, 512, 128)
    assert am0.ndim == 4
    assert am0.dtype == np.dtype('u2')

    xgm_beam_x = run['SPB_XTD9_XGM/DOOCS/MAIN', 'beamPosition.ixPos.value']
    assert len(xgm_beam_x.files) == 2
    assert xgm_beam_x.section == 'CONTROL'
    assert xgm_beam_x.entry_shape == ()
    assert xgm_beam_x.ndim == 1
    assert xgm_beam_x.dtype == np.dtype('f4')

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
    assert len(sel2.files) == 0
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
