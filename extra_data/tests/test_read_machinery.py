import os
import os.path as osp
from unittest import mock

import numpy as np
import pytest
import xarray as xr

from extra_data import RunDirectory, by_id, by_index, read_machinery
from extra_data.read_machinery import select_train_ids


def test_find_proposal(tmpdir):
    prop_dir = osp.join(str(tmpdir), 'SPB', '201701', 'p002012')
    os.makedirs(prop_dir)

    with mock.patch.object(read_machinery, 'DATA_ROOT_DIR', str(tmpdir)):
        assert read_machinery.find_proposal('p002012') == prop_dir

        assert read_machinery.find_proposal(prop_dir) == prop_dir


def test_same_run(mock_spb_raw_run, mock_jungfrau_run, mock_scs_run):
    run_spb = RunDirectory(mock_spb_raw_run)
    run_jf = RunDirectory(mock_jungfrau_run)
    run_scs = RunDirectory(mock_scs_run)

    assert run_spb.is_single_run
    assert run_jf.is_single_run
    assert run_scs.is_single_run

    assert not read_machinery.same_run(run_spb, run_scs, run_jf)

    format_version = run_spb.files[0].format_version
    s1 = run_spb.select_trains(np.s_[:1])
    s2 = run_spb.select_trains(np.s_[:-1])
    s3 = run_spb.select_trains(np.s_[3])

    s4 = run_spb.select('SA1_XTD2_XGM/DOOCS/MAIN', '*')
    s5 = run_spb.select('SPB_IRU_CAM/CAM/SIDEMIC:daqOutput', '*')

    if run_spb.run_metadata()['dataFormatVersion'] != '0.5':
        assert read_machinery.same_run(s1, s2, s3)
        assert read_machinery.same_run(s4, s5)
    else:
        assert not read_machinery.same_run(s1, s2, s3)
        assert not read_machinery.same_run(s4, s5)

    # SourceData
    sd = run_spb['SA1_XTD2_XGM/DOOCS/MAIN']
    sd1 = sd.select_trains(np.s_[:1])
    sd2 = sd.select_trains(np.s_[-1:])
    assert sd.is_single_run
    if sd.run_metadata()['dataFormatVersion'] != '0.5':
        assert read_machinery.same_run(sd1, sd2)
    else:
        assert not read_machinery.same_run(sd1, sd2)


def test_select_train_ids():
    train_ids = list(range(1000000, 1000010))

    # Test by_id with a single integer
    assert select_train_ids(train_ids, by_id[1000002]) == [1000002]

    # Test by_id with a numpy type
    assert select_train_ids(train_ids, by_id[np.uint64(1000002)]) == [1000002]

    # Test by_id with a slice
    assert select_train_ids(train_ids, by_id[1000002:1000005]) == [1000002, 1000003, 1000004]

    # Test by_id with a list
    assert select_train_ids(train_ids, by_id[[1000002, 1000005]]) == [1000002, 1000005]

    # Test by_id with a numpy array
    assert select_train_ids(train_ids, by_id[np.array([1000002, 1000005])]) == [1000002, 1000005]

    # Test by_id with a slice and step
    assert select_train_ids(train_ids, by_id[1000000:1000008:2]) == [1000000, 1000002, 1000004, 1000006]

    # Test by_id with an open-ended slice (end)
    assert select_train_ids(train_ids, by_id[1000005:]) == [1000005, 1000006, 1000007, 1000008, 1000009]

    # Test by_id with an open-ended slice (start)
    assert select_train_ids(train_ids, by_id[:1000003]) == [1000000, 1000001, 1000002]

    # Test by_index with a single integer
    assert select_train_ids(train_ids, by_index[2]) == [1000002]

    # Test by_index with a 0D numpy array
    assert select_train_ids(train_ids, by_index[np.array(5, dtype=np.int8)]) == [1000005]

    # Test by_index with a slice
    assert select_train_ids(train_ids, by_index[1:4]) == [1000001, 1000002, 1000003]

    # Test by_index with a list
    assert select_train_ids(train_ids, by_index[[1, 3]]) == [1000001, 1000003]

    # Test by_index with a slice and step
    assert select_train_ids(train_ids, by_index[::2]) == [1000000, 1000002, 1000004, 1000006, 1000008]

    # Test with a plain integer
    assert select_train_ids(train_ids, 3) == [1000003]

    # Test with a plain int-like
    assert select_train_ids(train_ids, np.uint32(3)) == [1000003]

    # Test with a plain slice
    assert select_train_ids(train_ids, slice(1, 4)) == [1000001, 1000002, 1000003]

    # Test with a plain list
    assert select_train_ids(train_ids, [1, 3]) == [1000001, 1000003]

    # Test with a numpy array
    assert select_train_ids(train_ids, np.array([1, 3])) == [1000001, 1000003]

    # Test with an invalid type (should raise TypeError)
    with pytest.raises(TypeError):
        select_train_ids(train_ids, "invalid")
    
    with pytest.raises(TypeError):
        select_train_ids(train_ids, by_id[np.float64(1000006)])

    # Test by_id with train IDs not found in the list (should raise a warning)
    with pytest.warns(UserWarning):
        result = select_train_ids(train_ids, by_id[[999999, 1000010]])
    assert result == []
    
    with pytest.warns(UserWarning):
        result = select_train_ids(train_ids, by_id[1000010])
    assert result == []

    # Test with boolean numpy array
    bool_array = np.zeros(len(train_ids), dtype=np.bool_)
    bool_array[[1, 3, 5]] = True
    assert select_train_ids(train_ids, bool_array) == [1000001, 1000003, 1000005]

    # Test with boolean numpy array of wrong length
    bool_array_wrong_size = np.ones(len(train_ids) + 2, dtype=np.bool_)
    with pytest.raises(ValueError, match="Boolean selector length"):
        select_train_ids(train_ids, bool_array_wrong_size)


def test_select_train_ids_xarray():
    train_ids = list(range(1000000, 1000010))
    
    # Create a boolean xarray with trainId coordinate
    trainid_coord = np.array([1000001, 1000003, 1000005, 1000007, 1000009])
    data = np.ones(len(trainid_coord), dtype=bool)
    data[-1] = False
    xr_sel = xr.DataArray(data, coords={'trainId': trainid_coord}, dims=['trainId'])

    # Test filtering with xarray
    assert select_train_ids(train_ids, xr_sel) == [1000001, 1000003, 1000005, 1000007]

    # Test with non-boolean xarray
    non_bool_xr = xr.DataArray(np.ones(5, dtype=float), coords={'trainId': trainid_coord}, dims=['trainId'])
    with pytest.raises(TypeError, match="xarray selector must be boolean"):
        select_train_ids(train_ids, non_bool_xr)

    # Test with xarray missing trainId coordinate
    wrong_coord_xr = xr.DataArray(np.ones(5, dtype=bool), coords={'time': np.arange(5)}, dims=['time'])
    with pytest.raises(ValueError, match="xarray selector must have a 'trainId' coordinate"):
        select_train_ids(train_ids, wrong_coord_xr)
