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

    # Control keys can omit .value suffix, but .keys() will not list that.
    assert 'beamPosition.ixPos.value' in xgm
    assert 'beamPosition.ixPos' in xgm
    assert 'beamPosition.ixPos.value' in xgm.keys()
    assert 'beamPosition.ixPos.timestamp' in xgm.keys()
    assert 'beamPosition.ixPos' not in xgm.keys()
    assert xgm['beamPosition.ixPos.value'].dtype == np.dtype('f4')
    assert xgm['beamPosition.ixPos'].dtype == np.dtype('f4')

    # .keys(inc_timestamp=False) will give us only the name before '.value'
    assert 'beamPosition.ixPos.value' not in xgm.keys(inc_timestamps=False)
    assert 'beamPosition.ixPos.timestamp' not in xgm.keys(inc_timestamps=False)
    assert 'beamPosition.ixPos' in xgm.keys(inc_timestamps=False)

def test_select_keys(mock_spb_raw_run):
    run = RunDirectory(mock_spb_raw_run)
    xgm = run['SPB_XTD9_XGM/DOOCS/MAIN']

    # Select exact key
    xpos_key = 'beamPosition.ixPos.value'
    assert xgm.select_keys('beamPosition.ixPos.value').keys() == {xpos_key}
    assert xgm.select_keys('beamPosition.ixPos').keys() == {xpos_key}
    assert xgm.select_keys({'beamPosition.ixPos.value'}).keys() == {xpos_key}
    assert xgm.select_keys({'beamPosition.ixPos'}).keys() == {xpos_key}

    # Select all keys
    all_keys = xgm.keys()
    assert xgm.select_keys(set()).keys() == all_keys
    assert xgm.select_keys(None).keys() == all_keys
    assert xgm.select_keys('*').keys() == all_keys

    # Select keys with glob pattern
    beampos_keys = {
        'beamPosition.ixPos.value', 'beamPosition.ixPos.timestamp',
        'beamPosition.iyPos.value', 'beamPosition.iyPos.timestamp'
    }
    assert xgm.select_keys('beamPosition.*').keys() == beampos_keys
    assert xgm.select_keys('beamPosition.*').select_keys('*').keys() == beampos_keys
