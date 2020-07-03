import numpy as np

from extra_data import RunDirectory

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

    sel1 = xgm_beam_x.select_trains(np.s_[:20])
    assert sel1.shape == (20,)
    assert len(sel1.files) == 1

    # Empty selection
    sel2 = xgm_beam_x.select_trains(np.s_[80:])
    assert sel2.shape == (0,)
    assert len(sel2.files) == 0
    assert sel2.xarray().shape == (0,)

