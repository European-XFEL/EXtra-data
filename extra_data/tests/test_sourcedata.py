import numpy as np
import pytest

from extra_data import RunDirectory, by_id, by_index
from extra_data.exceptions import PropertyNameError, SourceNameError, NoDataError


def test_get_sourcedata(mock_spb_raw_run):
    run = RunDirectory(mock_spb_raw_run)
    am0 = run['SPB_DET_AGIPD1M-1/DET/0CH0:xtdf']
    assert len(am0.files) == 1
    assert am0.section == 'INSTRUMENT'
    assert am0.is_instrument
    assert not am0.is_control
    assert not am0.is_run_only
    assert am0.index_groups == {'header', 'detector', 'image', 'trailer'}

    xgm = run['SPB_XTD9_XGM/DOOCS/MAIN']
    assert len(xgm.files) == 2
    assert xgm.section == 'CONTROL'
    assert xgm.is_control
    assert not xgm.is_run_only
    assert not xgm.is_instrument
    assert xgm.index_groups == {''}


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

    # Recreate the run and xgm objects so we can test one_key() when the
    # FileAccess caches are empty.
    run = RunDirectory(mock_spb_raw_run)
    xgm = run["SPB_XTD9_XGM/DOOCS/MAIN"]

    # Make sure that one_key() does indeed return a valid key for
    # control/instrument sources.
    assert xgm.one_key() in xgm.keys()
    xgm_output = run['SPB_XTD9_XGM/DOOCS/MAIN:output']
    assert xgm_output.one_key() in xgm_output.keys()

    # Test one_key() with index group.
    am0 = run['SPB_DET_AGIPD1M-1/DET/0CH0:xtdf']
    assert am0.one_key('image').startswith('image.')

    with pytest.raises(ValueError):
        # Asking for a de-selected index group.
        assert am0.select_keys('header.*').one_key('image')

    with pytest.raises(ValueError):
        # Not an index group of this source.
        assert am0.one_key('data')


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

    sel = xgm[by_id[10020:10040]]
    assert sel.train_ids == list(range(10020, 10040))

    sel = xgm[by_index[:10]]
    assert sel.train_ids == list(range(10000, 10010))

    sel = xgm[10]
    assert sel.train_ids == [10010]

    sel = xgm[999:1000]
    assert sel.train_ids == []
    assert sel.keys() == xgm.keys()


def test_split_trains(mock_spb_raw_run):
    run = RunDirectory(mock_spb_raw_run)
    xgm = run['SPB_XTD9_XGM/DOOCS/MAIN']
    assert len(xgm.train_ids) == 64

    chunks = list(xgm.split_trains(3))
    assert len(chunks) == 3
    assert {len(c.train_ids) for c in chunks} == {21, 22}
    # The middle chunk spans across 2 files
    assert [len(c.files) for c in chunks] == [1, 2, 1]

    chunks = list(xgm.split_trains(3, trains_per_part=20))
    assert len(chunks) == 4
    assert {len(c.train_ids) for c in chunks} == {16}


def test_union(mock_spb_raw_run):
    run = RunDirectory(mock_spb_raw_run)
    xgm = run['SPB_XTD9_XGM/DOOCS/MAIN']
    am0 = run['SPB_DET_AGIPD1M-1/DET/0CH0:xtdf']

    sel = xgm.select_trains(np.s_[:10]).union(xgm.select_trains(np.s_[-10:]))
    assert sel.train_ids == list(range(10000, 10010)) + list(range(10054, 10064))

    with pytest.raises(ValueError):
        xgm.union(am0)

    sel = xgm.select_trains(np.s_[:10]) | xgm.select_trains(np.s_[-10:])
    assert sel.train_ids == list(range(10000, 10010)) + list(range(10054, 10064))

    sel = xgm.select_trains(np.s_[:10])
    sel |= xgm.select_trains(np.s_[-10:])
    assert sel.train_ids == list(range(10000, 10010)) + list(range(10054, 10064))


def test_run_value(mock_spb_raw_run):
    run = RunDirectory(mock_spb_raw_run)
    xgm = run['SPB_XTD9_XGM/DOOCS/MAIN']
    am0 = run['SPB_DET_AGIPD1M-1/DET/0CH0:xtdf']

    value = xgm.run_value('pulseEnergy.conversion.value')
    assert isinstance(value, np.float64)

    run_dict = xgm.run_values()
    assert 'pulseEnergy.conversion.value' in run_dict
    assert 'pulseEnergy.conversion.timestamp' in run_dict

    values_dict = xgm.run_values(inc_timestamps=False)
    assert 'pulseEnergy.conversion' in values_dict
    assert 'pulseEnergy.conversion.timestamp' not in values_dict

    with pytest.raises(ValueError):
        # no run values for instrument sources
        am0.run_values()


def test_device_class(mock_spb_raw_run):
    run = RunDirectory(mock_spb_raw_run)
    xgm_ctrl = run['SPB_XTD9_XGM/DOOCS/MAIN']
    assert xgm_ctrl.device_class == 'DoocsXGM'

    xgm_inst = run['SPB_XTD9_XGM/DOOCS/MAIN:output']
    assert xgm_inst.device_class is None


def test_euxfel_path_infos(mock_spb_raw_run):
    run = RunDirectory(mock_spb_raw_run)
    xgm = run['SPB_XTD9_XGM/DOOCS/MAIN']

    assert xgm.storage_class is None  # Not an EuXFEL path.
    assert xgm.data_category == 'RAW'
    assert xgm.aggregator == 'DA01'

    # Changed to preserve the behaviour of above, as using a voview
    # file with 0-len datasets anyway causes a return of None. It
    # therefore attempts to use the regular .files property instead,
    # either suceeding or failing as badly as it would with a voview.
    run = RunDirectory(mock_spb_raw_run).select_trains(np.s_[:0])
    xgm = run['SPB_XTD9_XGM/DOOCS/MAIN']
    assert xgm.storage_class is None
    assert xgm.aggregator == 'DA01'


@pytest.mark.parametrize('source', [
    'SPB_XTD9_XGM/DOOCS/MAIN',  # Control data.
    'SPB_IRU_CAM/CAM/SIDEMIC:daqOutput',  # Pipeline data.
    'SPB_DET_AGIPD1M-1/DET/0CH0:xtdf'  # XTDF data.
])
def test_data_counts_modes(mock_reduced_spb_proc_run, source):
    run = RunDirectory(mock_reduced_spb_proc_run)
    sd = run[source]

    import pandas as pd

    for index_group in [None, *sd.index_groups]:
        count1 = sd.data_counts(index_group=index_group)
        assert isinstance(count1, pd.Series)
        assert count1.index.tolist() == sd.train_ids

        count2 = sd.data_counts(labelled=False, index_group=index_group)
        assert isinstance(count2, np.ndarray)
        assert len(count2) == len(sd.train_ids)

        np.testing.assert_equal(count1, count2)


def test_data_counts_values(mock_reduced_spb_proc_run):
    run = RunDirectory(mock_reduced_spb_proc_run)

    # control data
    xgm = run['SPB_XTD9_XGM/DOOCS/MAIN']
    assert (xgm.data_counts().values == 1).all()

    with pytest.raises(ValueError):
        xgm.data_counts(index_group='data')

    # instrument data
    camera = run['SPB_IRU_CAM/CAM/SIDEMIC:daqOutput']
    assert (camera.data_counts().values == 1).all()

    with pytest.raises(ValueError):
        camera.data_counts(index_group='not-data')

    am0 = run['SPB_DET_AGIPD1M-1/DET/0CH0:xtdf']
    num_images = am0['image.data'].shape[0]
    assert am0.data_counts().values.sum() >= num_images
    assert am0.data_counts(index_group='image').values.sum() == num_images

    with pytest.raises(ValueError):
        am0.data_counts(index_group='preamble')


def test_drop_empty_trains(mock_reduced_spb_proc_run):
    run = RunDirectory(mock_reduced_spb_proc_run)
    am0 = run['SPB_DET_AGIPD1M-1/DET/0CH0:xtdf']

    # Compare all index groups with `require_any`.
    np.testing.assert_equal(
        am0.drop_empty_trains().train_ids,
        run.select(am0.source, '*', require_any=True).train_ids)

    # Compare one specific index group with `require_all`.
    np.testing.assert_equal(
        am0.drop_empty_trains(index_group='image').train_ids,
        run.select(am0.source, 'image.*', require_all=True).train_ids)

    with pytest.raises(ValueError):
        am0.drop_empty_trains(index_group='preamble')


def test_train_id_coordinates(mock_reduced_spb_proc_run):
    run = RunDirectory(mock_reduced_spb_proc_run)

    # control data.
    xgm = run['SPB_XTD9_XGM/DOOCS/MAIN']

    np.testing.assert_equal(
        xgm.train_id_coordinates(),
        xgm.train_id_coordinates(''))
    np.testing.assert_equal(
        xgm.train_id_coordinates(),
        xgm['pulseEnergy.conversion'].train_id_coordinates())

    with pytest.raises(ValueError):
        xgm.train_id_coordinates('data')

    # instrument data.
    camera = run['SPB_IRU_CAM/CAM/SIDEMIC:daqOutput']

    np.testing.assert_equal(
        camera.train_id_coordinates(),
        camera.train_id_coordinates('data'))
    np.testing.assert_equal(
        camera.train_id_coordinates(),
        camera['data.image.pixels'].train_id_coordinates())

    with pytest.raises(ValueError):
        camera.train_id_coordinates('image')

    # xtdf data.
    am0 = run['SPB_DET_AGIPD1M-1/DET/0CH0:xtdf']

    np.testing.assert_equal(
        am0.train_id_coordinates('header'),
        am0['header.pulseCount'].train_id_coordinates())

    np.testing.assert_equal(
        am0.train_id_coordinates('image'),
        am0['image.data'].train_id_coordinates())

    # Should fail due to multiple index groups with differing counts.
    with pytest.raises(ValueError):
        am0.train_id_coordinates()


def test_legacy_sourcedata(mock_modern_spb_proc_run):
    run = RunDirectory(mock_modern_spb_proc_run)

    det_mod0 = 'SPB_DET_AGIPD1M-1/DET/0CH0:xtdf'
    corr_mod0 = 'SPB_DET_AGIPD1M-1/CORR/0CH0:output'

    # True (canonical) source works as normal
    sd = run[corr_mod0]
    assert sd.canonical_name == corr_mod0
    assert not sd.is_legacy

    # Obtaining SourceData object via legacy name emits a warning.
    with pytest.warns(DeprecationWarning):
        sd = run[det_mod0]

    assert sd.source == det_mod0
    assert sd.canonical_name == corr_mod0
    assert sd.is_legacy


def test_no_control_keys(mock_remi_run):
    run = RunDirectory(mock_remi_run)
    sd = run['SQS_REMI_DLD6/DET/TOP']

    assert sd.is_control
    assert sd.is_run_only
    assert not sd.keys()
    assert sd.one_key() is None
    assert sd.aggregator == 'REMI01'
    np.testing.assert_array_equal(sd.data_counts(), 0)
