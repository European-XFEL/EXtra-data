import numpy as np

from extra_data import RunDirectory

def test_get_sourcedata(mock_spb_raw_run):
    run = RunDirectory(mock_spb_raw_run)
    am0 = run['SPB_DET_AGIPD1M-1/DET/0CH0:xtdf']
    assert len(am0.files) == 1
    assert am0.section == 'INSTRUMENT'

    xgm = run['SPB_XTD9_XGM/DOOCS/MAIN']
    assert len(xgm.files) == 2
    assert xgm.section == 'CONTROL'

def test_keys(mock_spb_raw_run):
    run = RunDirectory(mock_spb_raw_run)
    xgm = run['SPB_XTD9_XGM/DOOCS/MAIN']

    assert 'beamPosition.ixPos.value' in xgm
    assert 'beamPosition.ixPos.value' in xgm.keys()
    assert xgm['beamPosition.ixPos.value'].dtype == np.dtype('f4')
