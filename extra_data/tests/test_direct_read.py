import numpy as np
import pytest

from extra_data import H5File, RunDirectory, direct_read, voview

from . import make_examples

SRC = 'SPB_DET_AGIPD1M-1/DET/0CH0:xtdf'


def assert_direct(kd, roi=(), dtype=None, threads=4):
    """Read *kd* with the direct reader, and check it against h5py."""
    if not isinstance(roi, tuple):
        roi = (roi,)

    expected = kd.ndarray(roi=roi, parallel=0)
    out = np.zeros(expected.shape, dtype or expected.dtype)

    assert direct_read.read(out, kd._read_ops(), roi, parallel=threads)
    np.testing.assert_array_equal(out, expected.astype(out.dtype))


@pytest.mark.parametrize('key', ['image.data', 'image.mask', 'image.gain'])
def test_layouts(mock_small_agipd_proc_run, key):
    # Uncompressed chunked, gzip chunked, and contiguous, respectively
    assert_direct(RunDirectory(mock_small_agipd_proc_run)[SRC, key])


@pytest.mark.parametrize('threads', [1, 4, 32])
def test_thread_counts(mock_small_agipd_proc_run, threads):
    assert_direct(RunDirectory(mock_small_agipd_proc_run)[SRC, 'image.data'], threads=threads)


@pytest.mark.parametrize('roi', [
    np.s_[8:24],        # A band of rows
    np.s_[:16, :8],     # Part of each row as well
    np.s_[::4],         # Strided
    np.s_[5],           # A single row, dropping that dimension
    np.s_[[1, 3, 7]],   # Arbitrary rows
    np.s_[:, 2:6],      # Columns only, so whole rows are read
])
@pytest.mark.parametrize('key', ['image.data', 'image.mask'])
def test_roi(mock_small_agipd_proc_run, key, roi):
    assert_direct(RunDirectory(mock_small_agipd_proc_run)[SRC, key], roi=roi)


@pytest.mark.parametrize('key', ['image.data', 'image.mask'])
def test_dtype_conversion(mock_small_agipd_proc_run, key):
    assert_direct(RunDirectory(mock_small_agipd_proc_run)[SRC, key], dtype=np.float64)


@pytest.mark.parametrize('key', ['image.data', 'image.mask'])
def test_train_selection(mock_small_agipd_proc_run, key):
    # Scattered trains give many separate reads; neighbouring ones can share an
    # HDF5 chunk, which should only be decompressed once.
    kd = RunDirectory(mock_small_agipd_proc_run)[SRC, key].select_trains(np.s_[::3])
    assert_direct(kd)
    assert_direct(kd, roi=np.s_[:16])


def test_ndarray_uses_it(mock_small_agipd_proc_run):
    kd = RunDirectory(mock_small_agipd_proc_run)[SRC, 'image.mask']
    np.testing.assert_array_equal(kd.ndarray(parallel=4),
                                  kd.ndarray(parallel=0))


def test_virtual_overview(tmp_path):
    # Its own run, because this test closes the files
    make_examples.make_small_agipd_proc_run(tmp_path)
    overview = tmp_path / 'overview.h5'
    run = RunDirectory(tmp_path)
    voview.VirtualOverviewFileWriter(overview, run).write()

    # HDF5 won't resolve a VDS mapping onto a file this process has open, so
    # the parallel=0 reads in assert_direct need these closed. The direct
    # reader resolves mappings itself.
    for file in run.files:
        file.close()
    del run

    kd = H5File(overview)[SRC, 'image.data']
    assert kd.files[0].file[kd.hdf5_data_path].is_virtual
    assert_direct(kd)
    assert_direct(kd, roi=np.s_[:16, :8])
    assert_direct(kd.select_trains(np.s_[::3]))


def test_falls_back_without_data(mock_small_agipd_proc_run):
    # A dataset with no data in it has no chunks to read, so h5py handles it
    kd = RunDirectory(mock_small_agipd_proc_run)[SRC, 'image.length']
    out = np.zeros(kd.shape, kd.dtype)

    assert not direct_read.read(out, kd._read_ops())
    np.testing.assert_array_equal(out, kd.ndarray(parallel=0))

    # Requiring the direct reader throws
    with pytest.raises(direct_read.UnsupportedDataset, match='not allocated'):
        kd.ndarray(parallel=4)


@pytest.mark.parametrize('kwargs', [
    dict(),
    dict(roi=np.s_[:16, :8]),
    dict(module_gaps=True),
    dict(astype=np.float64),
])
def test_multimod_detector(mock_small_agipd_proc_run, kwargs):
    # Detector data is read for all of the modules in one pass
    from extra_data.components import AGIPD1M

    det = AGIPD1M(RunDirectory(mock_small_agipd_proc_run), raw=False)
    for kd in [det['image.data'], det['image.data'].select_pulses(np.s_[::2])]:
        np.testing.assert_array_equal(kd.ndarray(parallel=4, **kwargs),
                                      kd.ndarray(parallel=0, **kwargs))
