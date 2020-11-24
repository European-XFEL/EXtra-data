import gc
import os
import pytest

from extra_data import file_access
from extra_data.reader import DataCollection

@pytest.fixture
def files_limit_512():
    orig_limiter = file_access.open_files_limiter
    file_access.open_files_limiter = l = file_access.OpenFilesLimiter(512)
    yield l
    file_access.open_files_limiter = orig_limiter

@pytest.fixture
def files_limit_3():
    orig_limiter = file_access.open_files_limiter
    file_access.open_files_limiter = l = file_access.OpenFilesLimiter(3)
    yield l
    file_access.open_files_limiter = orig_limiter


def test_filecache_large(mock_spb_raw_run, files_limit_512):
    fc = files_limit_512

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

def test_filecache_small(mock_spb_raw_run, files_limit_3):
    fc = files_limit_3

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

