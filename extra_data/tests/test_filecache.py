import gc
import os
import pytest
from collections import OrderedDict
from extra_data.reader import DataCollection

@pytest.fixture
def filecache_512():
    from extra_data.filecache import extra_data_filecache
    orig_cache = extra_data_filecache._cache
    orig_maxfiles = extra_data_filecache.maxfiles
    extra_data_filecache._cache = OrderedDict()
    extra_data_filecache._maxfiles = 512
    yield extra_data_filecache
    extra_data_filecache._cache = orig_cache
    extra_data_filecache._maxfiles = orig_maxfiles

@pytest.fixture
def filecache_3():
    from extra_data.filecache import extra_data_filecache
    orig_cache = extra_data_filecache._cache
    orig_maxfiles = extra_data_filecache.maxfiles
    extra_data_filecache._cache = OrderedDict()
    extra_data_filecache._maxfiles = 3
    yield extra_data_filecache
    extra_data_filecache._cache = orig_cache
    extra_data_filecache._maxfiles = orig_maxfiles


def test_filecache_large(mock_spb_raw_run, filecache_512):
    fc = filecache_512

    files = [os.path.join(mock_spb_raw_run, f) \
             for f in os.listdir(mock_spb_raw_run) if f.endswith('.h5')]
    run = DataCollection.from_paths(files)

    trains_iter = run.trains()
    tid, data = next(trains_iter)
    assert tid == 10000
    device = 'SPB_IRU_CAM/CAM/SIDEMIC:daqOutput'
    assert device in data
    assert data[device]['data.image.pixels'].shape == (1024, 768)
    # 16 AGIPD files + 1st DA file, but the other sequence file may be opened
    assert fc.n_open_files() >= 17

    del run, trains_iter
    gc.collect()
    assert fc.n_open_files() == 0

def test_filecache_small(mock_spb_raw_run, filecache_3):
    fc = filecache_3

    files = [os.path.join(mock_spb_raw_run, f) \
             for f in os.listdir(mock_spb_raw_run) if f.endswith('.h5')]
    run = DataCollection.from_paths(files)
    trains_iter = run.trains()

    for i in range(3):
        tid, data = next(trains_iter)
        assert tid == 10000 + i
        for j in range(16):
            device = f'SPB_DET_AGIPD1M-1/DET/{j}CH0:xtdf'
            assert device in data
            assert data[device]['image.data'].shape == (64, 2, 512, 128)
            assert len(fc._cache) == 3

    del run, trains_iter
    gc.collect()
    assert fc.n_open_files() == 0

