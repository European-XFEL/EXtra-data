import numpy as np
import pytest

from extra_data import RunDirectory, by_id, by_index
from extra_data.exceptions import PropertyNameError

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

    # select keys on INSTRUMENT data
    am0 = run['SPB_DET_AGIPD1M-1/DET/0CH0:xtdf']
    key = 'image.data'
    assert am0.select_keys(key).keys() == {key}
    assert am0.select_keys('*').keys() == am0.keys()

    with pytest.raises(PropertyNameError):
        am0.select_keys('data.image')

def test_select_trains(mock_spb_raw_run):
    run = RunDirectory(mock_spb_raw_run)
    xgm = run['SPB_XTD9_XGM/DOOCS/MAIN']

    assert len(xgm.train_ids) == 64
    sel = xgm.select_trains(by_id[10020:10040])
    assert sel.train_ids == list(range(10020, 10040))

    sel = xgm.select_trains(by_index[:10])
    assert sel.train_ids == list(range(10000, 10010))

    sel = xgm.select_trains(by_index[999995:999999])
    assert sel.train_ids == []
    assert sel.keys() == xgm.keys()

def test_union(mock_spb_raw_run):
    run = RunDirectory(mock_spb_raw_run)
    xgm = run['SPB_XTD9_XGM/DOOCS/MAIN']
    am0 = run['SPB_DET_AGIPD1M-1/DET/0CH0:xtdf']

    sel = xgm.select_trains(np.s_[:10]).union(xgm.select_trains(np.s_[-10:]))
    assert sel.train_ids == list(range(10000, 10010)) + list(range(10054, 10064))

    with pytest.raises(ValueError):
        xgm.union(am0)
