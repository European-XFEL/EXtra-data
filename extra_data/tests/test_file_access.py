import pickle

from ..file_access import FileAccess


def test_registry(mock_sa3_control_data):
    fa = FileAccess.get(mock_sa3_control_data)

    # Load some data to populate caches
    fa.get_index('SA3_XTD10_IMGFEL/CAM/BEAMVIEW2:daqOutput', 'data')
    fa.get_keys('SA3_XTD10_IMGFEL/CAM/BEAMVIEW2:daqOutput')

    assert len(fa._index_cache) == 1
    assert len(fa._keys_cache) == 1

    # This should get a reference to the existing object, not a duplicate.
    fa2 = FileAccess.get(mock_sa3_control_data)

    assert fa2 is fa

    # __init__() should not have been called again
    assert len(fa._index_cache) == 1
    assert len(fa._keys_cache) == 1


def test_pickle(mock_sa3_control_data):
    fa = FileAccess.get(mock_sa3_control_data)
    b = pickle.dumps(fa)

    # Load some data to populate caches
    fa.get_index('SA3_XTD10_IMGFEL/CAM/BEAMVIEW2:daqOutput', 'data')
    fa.get_keys('SA3_XTD10_IMGFEL/CAM/BEAMVIEW2:daqOutput')

    assert len(fa._index_cache) == 1
    assert len(fa._keys_cache) == 1

    # Unpickling should get a reference to the existing object, not a duplicate.
    fa2 = pickle.loads(b)

    assert fa2 is fa

    # Unpickling should not update state of existing object
    assert len(fa._index_cache) == 1
    assert len(fa._keys_cache) == 1
