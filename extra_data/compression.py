import h5py
import numpy as np
from zlib_into import decompress_into

def filter_ids(dset: h5py.Dataset):
    dcpl = dset.id.get_create_plist()
    return [dcpl.get_filter(i)[0] for i in range(dcpl.get_nfilters())]

class DeflateDecompressor:
    def __init__(self, deflate_filter_idx=0):
        self.deflate_filter_bit = (1 << deflate_filter_idx)

    @classmethod
    def for_dataset(cls, dset: h5py.Dataset):
        filters = filter_ids(dset)
        if filters == [h5py.h5z.FILTER_DEFLATE]:
            return cls()
        if dset.dtype.itemsize == 1 and filters == [
            h5py.h5z.FILTER_SHUFFLE, h5py.h5z.FILTER_DEFLATE
        ]:
            # The shuffle filter doesn't change single byte values, so we can
            # skip it.
            return cls(deflate_filter_idx=1)

    def apply_filters(self, data, filter_mask, out):
        if filter_mask & self.deflate_filter_bit:
            # The deflate filter is skipped, so just copy the data
            memoryview(out)[:] = data
        else:
            decompress_into(data, out)


class ShuffleDeflateDecompressor:
    def __init__(self, chunk_shape, dtype):
        chunk_nbytes = dtype.itemsize
        for l in chunk_shape:
            chunk_nbytes *= l
        # This will hold the decompressed data before shuffling
        self.chunk_buf = np.zeros(chunk_nbytes, dtype=np.uint8)
        self.shuffled_view = (     # E.g. for int32 data with chunks (10, 5):
                self.chunk_buf                  # (200,) uint8
                .reshape((dtype.itemsize, -1))  # (4, 50)
                .transpose()                    # (50, 4)
        )
        # Check this is still a view on the buffered data
        assert self.shuffled_view.base is self.chunk_buf

    @classmethod
    def for_dataset(cls, dset: h5py.Dataset):
        if filter_ids(dset) == [h5py.h5z.FILTER_SHUFFLE, h5py.h5z.FILTER_DEFLATE]:
            return cls(dset.chunks, dset.dtype)

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
            # Numpy does the shuffling by copying data between views
            out.reshape((-1, 1)).view(np.uint8)[:] = self.shuffled_view

from multiprocessing.pool import ThreadPool
from threading import Thread, local
from queue import Queue


def load_v1(kd, threads):
    # Read & decompress in each thread
    out = np.empty(kd.shape, dtype=kd.dtype)

    dset_ix_out_ix = []
    dest_cursor = 0
    for chunk in kd._data_chunks_nonempty:
        for i in range(chunk.total_count):
            dset_ix_out_ix.append((
                chunk.dataset, (chunk.first + i, 0, 0), out[dest_cursor + i]
            ))

        dest_cursor += chunk.total_count

    tlocal = local()

    def load_frame(dset, ds_ix, out):
        try:
            decomp = tlocal.decompressor
        except AttributeError:
            tlocal.decompressor = decomp = ShuffleDeflateDecompressor.for_dataset(dset)

        filter_mask, compdata = dset.id.read_direct_chunk(ds_ix)
        decomp.apply_filters(compdata, filter_mask, out)

    with ThreadPool(threads) as pool:
        pool.starmap(load_frame, dset_ix_out_ix)

    return out


def load_v2(kd, threads):
    # Read in the main thread, decompress in workers
    out = np.empty(kd.shape, dtype=kd.dtype)
    data_q = Queue(1024)

    workers = [DecompressorThread(data_q, kd._data_chunks[0].dataset) for _ in range(threads)]
    for w in workers:
        w.start()

    dest_cursor = 0
    for chunk in kd._data_chunks_nonempty:
        for i in range(chunk.total_count):
            filter_mask, compdata = chunk.dataset.id.read_direct_chunk((chunk.first + i, 0, 0))
            data_q.put((compdata, filter_mask, out[dest_cursor + i]))

        dest_cursor += chunk.total_count

    for _ in workers:
        data_q.put(None)  # Time to finish

    for w in workers:
        w.join(timeout=5)


class DecompressorThread(Thread):
    def __init__(self, input_q, eg_dset):
        super().__init__(daemon=True)
        self.input_q = input_q
        self.decomp = ShuffleDeflateDecompressor.for_dataset(eg_dset)

    def run(self):
        while True:
            if (recv := self.input_q.get()) is None:
                return  # Finished

            compdata, filter_mask, out = recv
            self.decomp.apply_filters(compdata, filter_mask, out)
