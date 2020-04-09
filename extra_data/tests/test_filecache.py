import os
import pytest
from collections import OrderedDict
from extra_data.reader import DataCollection

@pytest.fixture
def filecache_512():
    from extra_data import filecache
    orig_cache = filecache.extra_data_filecache
    filecache.extra_data_filecache = fc = filecache.FileCache(512)
    yield fc
    filecache.extra_data_filecache = orig_cache

@pytest.fixture
def filecache_3():
    from extra_data import filecache
    orig_cache = filecache.extra_data_filecache
    filecache.extra_data_filecache = fc = filecache.FileCache(3)
    yield fc
    filecache.extra_data_filecache = orig_cache


def test_filecache_large(mock_spb_raw_run, filecache_512):
    fc = filecache_512

    files = [os.path.join(mock_spb_raw_run, f) \
             for f in os.listdir(mock_spb_raw_run) if f.endswith('.h5')]
    run = DataCollection.from_paths(files)
    
    # estimate the number of files which should be open to read the first train
    cnt, fix = {}, {}
    for f in run.files:
        a, b = os.path.basename(f.filename).rsplit('-',1)
        try:
            cnt[a] += 1
        except KeyError:
            cnt[a] = 1
        if b == 'S00000.h5':
            fix[a] = cnt[a] 
    nfiles = sum(a for a in fix.values())

    trains_iter = run.trains()
    tid, data = next(trains_iter)
    assert tid == 10000
    device = 'SPB_IRU_CAM/CAM/SIDEMIC:daqOutput'
    assert device in data
    assert data[device]['data.image.pixels'].shape == (1024, 768)
    assert len(fc._cache) == nfiles
    
    del run, trains_iter
    assert len(fc._cache) == 0

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
    assert len(fc._cache) == 0

