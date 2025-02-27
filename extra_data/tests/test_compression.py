"""Test the decompression machinery"""

import h5py
import numpy as np

from extra_data.compression import (
    DeflateDecompressor, ShuffleDeflateDecompressor,
    dataset_decompressor, multi_dataset_decompressor, filter_ids,
)

def test_deflate(tmp_path):
    f = h5py.File(tmp_path / 'test.h5', 'w')
    # Shuffling single-byte data is a no-op
    arr = np.arange(200, dtype=np.uint8).reshape(4, 50)
    ds = f.create_dataset('d', data=arr, chunks=(4, 10), shuffle=True, compression='gzip')
    assert filter_ids(ds) == [h5py.h5z.FILTER_SHUFFLE, h5py.h5z.FILTER_DEFLATE]

    decomp = dataset_decompressor(ds)
    assert isinstance(decomp, DeflateDecompressor)

    filter_mask, data = ds.id.read_direct_chunk((0, 0))
    out = np.zeros((4, 10), dtype=np.uint8)
    decomp.apply_filters(data, filter_mask, out)
    np.testing.assert_array_equal(out, arr[:, :10])


def test_shuffle_deflate(tmp_path):
    f = h5py.File(tmp_path / 'test.h5', 'w')
    # Shuffling single-byte data is a no-op
    arr = np.arange(200, dtype=np.uint32).reshape(4, 50)
    ds = f.create_dataset('d', data=arr, chunks=(4, 10), shuffle=True, compression='gzip')
    assert filter_ids(ds) == [h5py.h5z.FILTER_SHUFFLE, h5py.h5z.FILTER_DEFLATE]

    decomp = dataset_decompressor(ds)
    assert isinstance(decomp, ShuffleDeflateDecompressor)

    filter_mask, data = ds.id.read_direct_chunk((0, 0))
    out = np.zeros((4, 10), dtype=np.uint32)
    decomp.apply_filters(data, filter_mask, out)
    np.testing.assert_array_equal(out, arr[:, :10])


def test_multi_dataset_decompressor(tmp_path):
    f = h5py.File(tmp_path / 'test.h5', 'w')
    ds1 = f.create_dataset('a', shape=(10, 50), chunks=(1, 50),
                           compression='gzip', dtype=np.uint32)
    ds2 = f.create_dataset('b', shape=(20, 50), chunks=(1, 50),
                           compression='gzip', dtype=np.uint32)
    ds3 = f.create_dataset('c', shape=(10, 50), chunks=(1, 50),
                           compression='gzip', dtype=np.uint8)

    # Differing shape is OK
    assert isinstance(multi_dataset_decompressor([ds1, ds2]), DeflateDecompressor)

    # But dtype needs to match
    assert multi_dataset_decompressor([ds1, ds3]) is None

    assert multi_dataset_decompressor([]) is None
