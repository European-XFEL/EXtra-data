"""Test the decompression machinery"""

import h5py
import numpy as np

from extra_data.compression import (
    DeflateDecompressor, ShuffleDeflateDecompressor,
    dataset_decompressor, filter_ids,
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

