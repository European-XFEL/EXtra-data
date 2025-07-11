import threading
from copy import copy
from multiprocessing.pool import ThreadPool

import h5py
import numpy as np
from zlib_into import decompress_into, unshuffle


def filter_ids(dset: h5py.Dataset):
    dcpl = dset.id.get_create_plist()
    return [dcpl.get_filter(i)[0] for i in range(dcpl.get_nfilters())]


class DeflateDecompressor:
    def __init__(self, deflate_filter_idx=0):
        self.deflate_filter_bit = 1 << deflate_filter_idx

    @classmethod
    def for_dataset(cls, dset: h5py.Dataset):
        filters = filter_ids(dset)
        if filters == [h5py.h5z.FILTER_DEFLATE]:
            return cls()
        if dset.dtype.itemsize == 1 and filters == [
            h5py.h5z.FILTER_SHUFFLE,
            h5py.h5z.FILTER_DEFLATE,
        ]:
            # The shuffle filter doesn't change single byte values, so we can
            # skip it.
            return cls(deflate_filter_idx=1)

        return None

    def clone(self):
        return copy(self)

    def apply_filters(self, data, filter_mask, out):
        if filter_mask & self.deflate_filter_bit:
            # The deflate filter is skipped, so just copy the data
            memoryview(out)[:] = data
        else:
            decompress_into(data, out)


class ShuffleDeflateDecompressor:
    def __init__(self, chunk_shape, dtype):
        self.chunk_shape = chunk_shape
        self.dtype = dtype
        chunk_nbytes = dtype.itemsize
        for l in chunk_shape:
            chunk_nbytes *= l
        # This will hold the decompressed data before unshuffling
        self.chunk_buf = np.zeros(chunk_nbytes, dtype=np.uint8)

    @classmethod
    def for_dataset(cls, dset: h5py.Dataset):
        if filter_ids(dset) == [h5py.h5z.FILTER_SHUFFLE, h5py.h5z.FILTER_DEFLATE]:
            return cls(dset.chunks, dset.dtype)

        return None

    def clone(self):
        return type(self)(self.chunk_shape, self.dtype)

    def apply_filters(self, data, filter_mask, out):
        if filter_mask & 2:
            # The deflate filter is skipped
            memoryview(self.chunk_buf)[:] = data
        else:
            decompress_into(data, self.chunk_buf)

        if filter_mask & 1:
            # The shuffle filter is skipped
            memoryview(out)[:] = self.chunk_buf
        else:
            unshuffle(self.chunk_buf, out, self.dtype.itemsize)


def dataset_decompressor(dset):
    for cls in [DeflateDecompressor, ShuffleDeflateDecompressor]:
        if (inst := cls.for_dataset(dset)) is not None:
            return inst

    return None


def multi_dataset_decompressor(dsets):
    if not dsets:
        return None

    chunk = dsets[0].chunks
    dtype = dsets[0]
    filters = filter_ids(dsets[0])
    for d in dsets[1:]:
        if d.chunks != chunk or d.dtype != dtype or filter_ids(d) != filters:
            return None  # Datasets are not consistent

    return dataset_decompressor(dsets[0])


def parallel_decompress_chunks(tasks, decompressor_proto, threads=16):
    tlocal = threading.local()

    def load_one(dset_id, coord, dest):
        try:
            decomp = tlocal.decompressor
        except AttributeError:
            tlocal.decompressor = decomp = decompressor_proto.clone()

        if dset_id.get_chunk_info_by_coord(coord).byte_offset is None:
            return   # Chunk not allocated in file

        filter_mask, compdata = dset_id.read_direct_chunk(coord)
        decomp.apply_filters(compdata, filter_mask, dest)

    with ThreadPool(threads) as pool:
        pool.starmap(load_one, tasks)
