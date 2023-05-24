import gc
import pickle

import cloudpickle
import pytest

import numpy as np

from ..file_access import FileAccess
from ..exceptions import FileStructureError


def test_registry(mock_sa3_control_data):
    fa = FileAccess(mock_sa3_control_data)

    # Load some data to populate caches
    fa.get_index('SA3_XTD10_IMGFEL/CAM/BEAMVIEW2:daqOutput', 'data')
    fa.get_keys('SA3_XTD10_IMGFEL/CAM/BEAMVIEW2:daqOutput')

    assert len(fa._index_cache) == 1
    assert len(fa._keys_cache) == 1

    # This should get a reference to the existing object, not a duplicate.
    fa2 = FileAccess(mock_sa3_control_data)

    assert fa2 is fa

    # __init__() should not have been called again
    assert len(fa._index_cache) == 1
    assert len(fa._keys_cache) == 1


@pytest.mark.parametrize('pickle_mod', (pickle, cloudpickle))
def test_pickle(pickle_mod, mock_sa3_control_data):
    fa = FileAccess(mock_sa3_control_data)
    b = pickle_mod.dumps(fa)

    # Load some data to populate caches
    fa.get_index('SA3_XTD10_IMGFEL/CAM/BEAMVIEW2:daqOutput', 'data')
    fa.get_keys('SA3_XTD10_IMGFEL/CAM/BEAMVIEW2:daqOutput')

    assert len(fa._index_cache) == 1
    assert len(fa._keys_cache) == 1

    # Unpickling should get a reference to the existing object, not a duplicate.
    fa2 = pickle_mod.loads(b)

    assert fa2 is fa

    # Unpickling should not update state of existing object
    assert len(fa._index_cache) == 1
    assert len(fa._keys_cache) == 1

    # Delete the existing instances, then reload from pickle
    del fa, fa2
    gc.collect()

    fa3 = pickle_mod.loads(b)
    assert len(fa3._index_cache) == 0
    assert len(fa3._keys_cache) == 0
    assert 'SA3_XTD10_IMGFEL/CAM/BEAMVIEW2:daqOutput' in fa3.instrument_sources
    assert len(fa3.train_ids) == 500


# Empty FileAccess cache entry to test behaviour without actually trying
# to read a non-existing file in tests below.
_empty_cache_info = dict(
    train_ids= np.zeros(0, dtype=np.uint64),
    control_sources=frozenset(),
    instrument_sources=frozenset(),
    flag=np.zeros(0, dtype=np.int32)
)


def test_euxfel_path_infos(mock_sa3_control_data, monkeypatch):
    fa = FileAccess(mock_sa3_control_data)

    # Default path is not a EuXFEL storage location.
    assert fa.storage_class is None
    assert fa.instrument is None
    assert fa.cycle is None

    # EuXFEL locations are resolved to their true paths and may either
    # be on online GPFS, offline GPFS or dCache.
    for filename in [
        '/gpfs/exfel/exp/SA3/202301/p001234/raw/r0100/foo.h5',
        '/gpfs/exfel/d/raw/SA3/202301/p001234/r0100/foo.h5',
        '/pnfs/xfel.eu/exfel/archive/XFEL/raw/SA3/202301/p001234/r0100/foo.h5'
    ]:
        fa = FileAccess(filename, _cache_info=_empty_cache_info)

        assert fa.storage_class == 'raw'
        assert fa.instrument == 'SA3'
        assert fa.cycle == '202301'


def test_euxfel_filename_infos(mock_sa3_control_data, monkeypatch):
    fa = FileAccess(mock_sa3_control_data)

    assert fa.data_category == 'RAW'
    assert fa.aggregator == 'DA01'
    assert fa.sequence == 1

    fa = FileAccess('/a/b/c/my-own-file.h5', _cache_info=_empty_cache_info)

    assert fa.data_category is None
    assert fa.aggregator is None
    assert fa.sequence is None


def test_no_index(empty_h5_file):
    with pytest.raises(FileStructureError):
        FileAccess(empty_h5_file)


def test_no_metadata(mock_no_metadata_file):
    with pytest.raises(FileStructureError):
        FileAccess(mock_no_metadata_file)
