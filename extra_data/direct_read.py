import os
import os.path as osp
import sys
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np

from .utils import default_num_threads

# Size to aim for per read of uncompressed data. Anything from 4 to 44 MiB is
# the same speed, but on GPFS a request of 48 MiB or more stops overlapping with
# the other threads, dropping to what a single thread gets whatever the thread
# count (~40 -> ~5 GB/s at 16 threads). 32MiB seems like a nice big number.
SPLIT_BYTES = 32 * 1024 ** 2

# Compressed reads are decompression-bound and get slower past ~16
# threads. Unfiltered reads are flat from 16 when cold and gain ~10% warm up to
# ~48. Capped by core count.
THREADS_FILTERED = 16
THREADS_UNFILTERED = 48


class UnsupportedDataset(Exception):
    """Raised while planning, for a dataset we can't read ourselves."""


def read_debug(msg, *args):
    """Say what a read did, if EXTRA_DATA_DIRECT_READ_DEBUG is set.

    Whether a read falls back to h5py is otherwise invisible. The environment
    is checked each time, so this can be switched on part way through a session.
    """
    if os.environ.get('EXTRA_DATA_DIRECT_READ_DEBUG'):
        print('extra_data.direct_read:', msg % args, file=sys.stderr)


# What's in the file: ---------------------------------------------------------

def sel_block(space):
    """The solid rectangle a dataspace selects: (starts, ends, shape).

    `ends` are exclusive. Raises UnsupportedDataset if the selection isn't one solid
    rectangle. The number of blocks says nothing about that: h5py builds these
    selections out of one unit block per element.
    """
    shape = space.get_simple_extent_dims()

    if space.get_select_type() == h5py.h5s.SEL_ALL:
        return (0,) * len(shape), tuple(shape), tuple(shape)

    if not space.is_regular_hyperslab():
        raise UnsupportedDataset("virtual mapping is not a regular hyperslab")

    starts, strides, counts, blocks = space.get_regular_hyperslab()
    ends = []
    for start, stride, count, block in zip(starts, strides, counts, blocks):
        if count > 1 and stride != block:
            raise UnsupportedDataset("virtual mapping has gaps in it")
        ends.append(int(start) + int(count) * int(block))

    return tuple(int(i) for i in starts), tuple(ends), tuple(shape)


def virtual_maps(dset):
    """Where a virtual dataset gets its data from.

    Returns a list of ``(start, stop, filename, dataset_path, src_start)``,
    sorted by `start`, saying that entries ``[start:stop]`` of this dataset are
    entries ``[src_start:src_start + stop - start]`` of another one.

    Only mappings that shift a contiguous run of entries along the first
    dimension are understood, which is all EXtra-data itself makes.
    """
    dirname = osp.dirname(dset.file.filename)
    maps = []

    for vspace, filename, ds_path, src_space in dset.virtual_sources():
        v_start, v_end, v_shape = sel_block(vspace)
        s_start, s_end, s_shape = sel_block(src_space)

        # Dimensions after the first must map straight through, in full.
        if v_shape[1:] != s_shape[1:]:
            raise UnsupportedDataset("virtual mapping changes the entry shape")
        for start, end, length in zip(v_start[1:] + s_start[1:],
                                      v_end[1:] + s_end[1:],
                                      v_shape[1:] + s_shape[1:]):
            if start != 0 or end != length:
                raise UnsupportedDataset("virtual mapping covers part of an entry")

        if (v_end[0] - v_start[0]) != (s_end[0] - s_start[0]):
            raise UnsupportedDataset("virtual mapping is not one-to-one")

        maps.append((v_start[0], v_end[0], osp.join(dirname, filename),
                     ds_path, s_start[0]))

    maps.sort()
    return maps


def get_decompressor(dset):
    """A prototype decompressor for this dataset, or None if unfiltered."""
    # Counting filters needs only h5py, so uncompressed data can be read
    # without the optional decompressors installed.
    if dset.id.get_create_plist().get_nfilters() == 0:
        return None

    try:
        from .compression import dataset_decompressor
    except ImportError as e:
        raise UnsupportedDataset(f"can't load the decompressors ({e})")

    decompressor = dataset_decompressor(dset)
    if decompressor is None:
        raise UnsupportedDataset("no fast decompressor for these filters")

    return decompressor


class DatasetInfo:
    """What we need to know about one real (non-virtual) dataset.

    The chunk table isn't cheap to read, and never changes for a given file, so
    this is cached on the FileAccess.
    """
    def __init__(self, dset, ds_path):
        if dset.dtype.hasobject or dset.dtype.kind not in 'biufc':
            raise UnsupportedDataset(f"dtype {dset.dtype} is not a plain number")
        if dset.chunks is not None and dset.chunks[1:] != dset.shape[1:]:
            # We rely on each chunk holding whole entries, contiguously.
            raise UnsupportedDataset("chunks don't cover whole entries")

        self.filename = dset.file.filename
        self.key = (self.filename, ds_path)  # Identifies it in dicts
        self.dtype = dset.dtype
        self.shape = dset.shape
        self.entry_shape = dset.shape[1:]
        self.frame_bytes = self.dtype.itemsize * int(
            np.prod(self.entry_shape, dtype=np.intp))
        self.decompressor = get_decompressor(dset)

        # {first entry of chunk: (byte offset, stored size, filter mask)}
        if dset.chunks is None:
            # Unchunked datasets are treated as a single unfiltered chunk
            self.chunk_frames = max(1, dset.shape[0])
            offset = dset.id.get_offset()
            if offset is None:
                raise UnsupportedDataset("dataset has no data in the file")
            self.chunks = {0: (offset, self.chunk_frames * self.frame_bytes, 0)}
        else:
            self.chunk_frames = dset.chunks[0]
            self.chunks = {}
            dset.id.chunk_iter(lambda info: self.chunks.__setitem__(
                info.chunk_offset[0],
                (info.byte_offset, info.size, info.filter_mask)
            ))

        self.chunk_nbytes = self.chunk_frames * self.frame_bytes

    def chunk_first(self, entry):
        """Where the chunk holding `entry` starts."""
        return (entry // self.chunk_frames) * self.chunk_frames

    def chunk_of(self, entry):
        """Where the chunk holding `entry` starts, and what's in the table."""
        first = self.chunk_first(entry)
        try:
            return first, self.chunks[first]
        except KeyError:
            # HDF5 would give the fill value here, we don't support that
            raise UnsupportedDataset("chunk is not allocated in the file")


def band_index(roi_dim0, nrows):
    """Which rows of an entry to read for `roi_dim0`, and how to index them.

    Returns ``(first, stop, index)``: rows ``[first:stop]`` of each entry are
    read, and indexing those rows with `index` gives what the ROI asked for.
    """
    if isinstance(roi_dim0, slice):
        start, stop, step = roi_dim0.indices(nrows)
        if step > 0 and stop > start:
            last = start + ((stop - start - 1) // step) * step
            return start, last + 1, slice(0, last - start + 1, step)

    elif isinstance(roi_dim0, (int, np.integer)):
        row = int(roi_dim0)
        if row < 0:
            row += nrows
        if 0 <= row < nrows:
            return row, row + 1, 0

    elif isinstance(roi_dim0, (list, np.ndarray)):
        index = np.asarray(roi_dim0)
        if index.dtype.kind in 'iu' and index.size and index.min() >= 0:
            first = int(index.min())
            return first, int(index.max()) + 1, index - first

    # Anything else (boolean mask, empty selection, negative step): read whole
    # entries and let numpy index them.
    return 0, nrows, roi_dim0


def noop_roi(roi, entry_shape):
    """Whether `roi` selects every element of an entry, unchanged."""
    return all(isinstance(r, slice) and r.indices(n) == (0, n, 1)
               for r, n in zip(roi, entry_shape))


def buffer(scratch, key, nbytes):
    """A byte buffer of exactly `nbytes`, re-used between jobs."""
    buf = scratch.get(key)
    if buf is None or buf.size < nbytes:
        buf = scratch[key] = np.empty(nbytes, dtype=np.uint8)

    return buf[:nbytes]


class DatasetReader:
    """A dataset to read from, with the details of one particular read.

    Dataset facts come from a cached :class:`DatasetInfo`; this adds the file
    descriptor, whatever depends on the ROI and the output array, and the reads.

    The descriptor is deliberately not cached: files get closed and reopened
    elsewhere, and a recycled fd number would read the wrong file rather than
    fail.
    """
    def __init__(self, info, fd, roi, out_dtype):
        self.info = info
        self.fd = fd

        if len(roi) > len(info.entry_shape):
            raise UnsupportedDataset("the ROI has more dimensions than an entry")

        # Padded to the entry's dimensions, so there's no separate no-ROI case.
        # Decompressing gives back whole entries, so that path applies this as
        # written; band reads use the band-relative version made below.
        self.roi = roi + (np.s_[:],) * (len(info.entry_shape) - len(roi))

        self._plan_band()

        # With whole entries and no conversion, the file's bytes are exactly
        # what belongs in the output array, so we can read straight into it.
        self.whole_frame = (self.band_bytes == info.frame_bytes)
        self.direct = (self.whole_frame and out_dtype == info.dtype
                       and noop_roi(self.roi, info.entry_shape))

        self.read_run = self._read_direct if self.direct else self._read_band

    def _plan_band(self):
        """Which rows of an entry the ROI needs, and how to index them.

        The rows it touches are read in full: rows are contiguous within an
        entry, columns are not.
        """
        info = self.info

        if info.entry_shape:
            first, stop, index = band_index(self.roi[0], info.entry_shape[0])
            self.band_shape = (stop - first,) + info.entry_shape[1:]
            self.band_index = (index,) + self.roi[1:]
        else:
            first = 0
            self.band_shape = ()
            self.band_index = ()

        self.band_offset = first * info.dtype.itemsize * int(
            np.prod(info.entry_shape[1:], dtype=np.intp))
        self.band_bytes = info.dtype.itemsize * int(
            np.prod(self.band_shape, dtype=np.intp))

    # Doing the reading: ------------------------------------------------------

    # A job is a (bound method, arguments) pair, called as
    # method(scratch, out, *args), and is always a single read. `scratch` holds
    # one worker's buffers and decompressors.

    def _pread(self, buf, offset):
        got = os.preadv(self.fd, [buf], offset)
        if got != buf.nbytes:
            raise EOFError(f"read {got} of {buf.nbytes} bytes at {offset} from "
                           f"{self.info.filename}")

    def _read_direct(self, scratch, out, offset, dest_first, count):
        """Copy a run of entries from the file straight into the output array."""
        dest = out[dest_first:dest_first + count]
        self._pread(dest.reshape(-1).view(np.uint8), offset)

    def _read_band(self, scratch, out, offset, dest_first, count):
        """Read part of each entry, or convert it, on the way into `out`.

        `count` is 1 unless whole entries are being read: only whole entries are
        contiguous with each other in the file.
        """
        buf = buffer(scratch, 'band', count * self.band_bytes)
        self._pread(buf, offset)

        data = buf.view(self.info.dtype).reshape((count,) + self.band_shape)
        out[dest_first:dest_first + count] = data[(np.s_[:],) + self.band_index]

    def _decompress(self, scratch, offset, size, filter_mask, dest):
        """Read one compressed chunk from the file and unpack it into `dest`."""
        compressed = buffer(scratch, 'compressed', size)
        self._pread(compressed, offset)

        # Decompressors hold a buffer, so each worker needs its own copy of the
        # prototype made while planning.
        key = ('decompressor', self.info.key)
        decompressor = scratch.get(key)
        if decompressor is None:
            decompressor = scratch[key] = self.info.decompressor.clone()

        decompressor.apply_filters(compressed, filter_mask, dest)

    def read_chunk(self, scratch, out, offset, size, filter_mask, segments):
        """Read one compressed chunk and unpack the wanted entries out of it.

        `segments` are ``(first entry in the chunk, count, destination index)``.
        A chunk is decompressed whole however little of it is wanted, so
        everything needed from one chunk is a single job.
        """
        chunk = buffer(scratch, 'chunk', self.info.chunk_nbytes)
        self._decompress(scratch, offset, size, filter_mask, chunk)

        data = chunk.view(self.info.dtype).reshape(
            (self.info.chunk_frames,) + self.info.entry_shape)

        for first, count, dest_first in segments:
            # These are whole entries, so the ROI applies as the caller wrote it
            out[dest_first:dest_first + count] = \
                data[first:first + count][(np.s_[:],) + self.roi]

    def read_chunk_inplace(self, scratch, out, offset, size, filter_mask,
                           dest_first):
        """Decompress a whole chunk into the output array, with no copy.

        Used where the chunk's entries all belong in one contiguous piece of
        `out`, exactly as they are in the file. That is the common case of
        reading everything.
        """
        dest = out[dest_first:dest_first + self.info.chunk_frames]
        self._decompress(scratch, offset, size, filter_mask,
                         dest.reshape(-1).view(np.uint8))


# Planning: -------------------------------------------------------------------

class Planner:
    """Turns a list of :class:`ReadOp` into a list of jobs.

    All the HDF5 work (resolving virtual datasets, reading chunk tables) happens
    here so that running the jobs doesn't need HDF5.
    """
    def __init__(self, out_dtype, roi):
        self.out_dtype = out_dtype
        self.roi = roi
        self.jobs = []
        self.fds = {}       # filename -> file descriptor
        self._files = {}    # filename -> h5py.File we opened and must close
        self._borrowed = {} # filename -> FileAccess whose file we may use
        self._readers = {}  # (filename, dataset path) -> DatasetReader

        # Which entries are wanted from each compressed chunk, turned into jobs
        # at the end: separate runs of entries can land in the same chunk, and
        # that chunk is only worth reading once.
        self._chunk_segments = {}  # (reader, first entry of chunk) -> segments

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self._close_files()
        for fd in self.fds.values():
            os.close(fd)
        self.fds.clear()

    def _close_files(self):
        """Close the HDF5 files, keeping the descriptors we read through."""
        for file in self._files.values():
            file.close()
        self._files.clear()
        self._borrowed.clear()

    def _dataset(self, filename, ds_path):
        """An h5py dataset, through a FileAccess where we have one.

        A borrowed file is fetched each time rather than held onto: the open
        file limiter may close it between ops.
        """
        file_access = self._borrowed.get(filename)
        if file_access is not None:
            file = file_access.file
        elif filename in self._files:
            file = self._files[filename]
        else:
            file = self._files[filename] = h5py.File(filename, 'r')

        try:
            return file[ds_path]
        except KeyError:
            raise UnsupportedDataset(f"{ds_path} not found in {filename}")

    def _fd(self, filename):
        fd = self.fds.get(filename)
        if fd is None:
            fd = self.fds[filename] = os.open(filename, os.O_RDONLY)

        return fd

    def _reader(self, cache, filename, ds_path):
        """The dataset at `ds_path`, which must not be a virtual dataset."""
        reader = self._readers.get((filename, ds_path))
        if reader is not None:
            return reader

        info = cache.get(('info', filename, ds_path))
        if info is None:
            dset = self._dataset(filename, ds_path)
            if dset.is_virtual:
                # One level only: overview files point at real files, and we
                # never nest them.
                raise UnsupportedDataset("virtual dataset points to another one")

            info = DatasetInfo(dset, ds_path)
            cache[('info', filename, ds_path)] = info

        reader = DatasetReader(info, self._fd(filename), self.roi, self.out_dtype)
        self._readers[(filename, ds_path)] = reader
        return reader

    def plan(self, ops):
        try:
            for op in ops:
                self._add_op(op)

            for (reader, chunk_first), segments in self._chunk_segments.items():
                self._add_chunk_job(reader, chunk_first, segments)
        finally:
            # Nothing from here on needs HDF5, only the file descriptors.
            self._close_files()

        return self.jobs

    def all_unfiltered(self):
        """True if none of the datasets we planned are compressed."""
        return all(r.info.decompressor is None for r in self._readers.values())

    def _add_op(self, op):
        """Plan one contiguous run of entries from one dataset."""
        # Chunk tables and virtual mappings are cached on the FileAccess, so
        # reading the same data again doesn't pay for them again.
        cache = op.file._direct_read_cache
        filename, ds_path = op.file.filename, op.dataset_path
        self._borrowed.setdefault(filename, op.file)

        dset = self._dataset(filename, ds_path)
        if not dset.is_virtual:
            reader = self._reader(cache, filename, ds_path)
            self._add(reader, op.src_first, op.dest_first, op.count)
            return

        maps = cache.get(('virtual', ds_path))
        if maps is None:
            maps = cache[('virtual', ds_path)] = virtual_maps(dset)

        # Sorted, so walking them in order fills the run from its start; a gap
        # means part of it maps to no real data.
        cursor, end = op.src_first, op.src_first + op.count
        for map_start, map_stop, map_file, map_path, map_src in maps:
            if map_stop <= cursor:
                continue
            if map_start > cursor or map_start >= end:
                break

            reader = self._reader(cache, map_file, map_path)
            count = min(map_stop, end) - cursor
            self._add(reader, map_src + (cursor - map_start),
                      op.dest_first + (cursor - op.src_first), count)

            cursor += count
            if cursor >= end:
                return

        raise UnsupportedDataset("virtual dataset doesn't map all the data")

    def _add(self, reader, src_first, dest_first, count):
        if count == 0:
            return

        if src_first + count > reader.info.shape[0]:
            raise UnsupportedDataset("read runs past the end of the dataset")

        if reader.info.decompressor is None:
            self._add_unfiltered(reader, src_first, dest_first, count)
        else:
            self._add_filtered(reader, src_first, dest_first, count)

    def _add_unfiltered(self, reader, src_first, dest_first, count):
        """Plan reads of stored-as-is data, which can be split anywhere.

        Entries are contiguous within a chunk, so a chunk needs at most one
        read. Long runs are split further, so that a few big chunks can still be
        read by many threads at once.
        """
        if reader.whole_frame:
            # A job covers at most SPLIT_BYTES, which bounds a worker's scratch.
            max_per_read = max(1, SPLIT_BYTES // max(1, reader.band_bytes))
        else:
            # Only whole entries are contiguous with each other in the file, so
            # a job reading part of an entry covers exactly one.
            max_per_read = 1

        cursor, end = src_first, src_first + count
        while cursor < end:
            chunk_first, (byte_offset, _, filter_mask) = \
                reader.info.chunk_of(cursor)
            if filter_mask:
                raise UnsupportedDataset("filters skipped on an unfiltered dataset")

            n = min(end, chunk_first + reader.info.chunk_frames) - cursor
            offset = (byte_offset
                      + ((cursor - chunk_first) * reader.info.frame_bytes)
                      + reader.band_offset)

            for start in range(0, n, max_per_read):
                self.jobs.append((reader.read_run, (
                    offset + (start * reader.info.frame_bytes),
                    dest_first + (cursor - src_first) + start,
                    min(max_per_read, n - start),
                )))

            cursor += n

    def _add_filtered(self, reader, src_first, dest_first, count):
        """Note which entries are wanted from each chunk of compressed data."""
        cursor, end = src_first, src_first + count
        while cursor < end:
            chunk_first = reader.info.chunk_first(cursor)
            n = min(end, chunk_first + reader.info.chunk_frames) - cursor

            self._chunk_segments.setdefault((reader, chunk_first), []).append(
                (cursor - chunk_first, n, dest_first + (cursor - src_first))
            )

            cursor += n

    def _add_chunk_job(self, reader, chunk_first, segments):
        _, (byte_offset, size, filter_mask) = reader.info.chunk_of(chunk_first)
        args = (byte_offset, size, filter_mask)

        # A whole chunk landing unchanged in one contiguous piece of the output
        # can be decompressed straight into it. The last chunk may hang over the
        # end of the dataset, where not all of it belongs there.
        first, count, dest_first = segments[0]
        if (len(segments) == 1 and reader.direct
                and count == reader.info.chunk_frames
                and chunk_first + count <= reader.info.shape[0]):
            self.jobs.append((reader.read_chunk_inplace, args + (dest_first,)))
        else:
            self.jobs.append((reader.read_chunk, args + (segments,)))


# Running the plan: -----------------------------------------------------------

def run_batch(jobs, out):
    scratch = {}
    for func, args in jobs:
        func(scratch, out, *args)


def run(jobs, out, threads):
    """Run the jobs, spread over `threads` workers.
    """
    n_batches = min(threads, len(jobs))
    if n_batches <= 1:
        run_batch(jobs, out)
        return

    # Submit batches of jobs to the pool
    size = -(-len(jobs) // n_batches)
    batches = [jobs[i * size:(i + 1) * size] for i in range(n_batches)]
    with ThreadPoolExecutor(n_batches) as pool:
        results = [pool.submit(run_batch, batch, out) for batch in batches]
        for result in results:
            result.result()


def read_directly(out, ops, roi, threads):
    """Fill `out` from `ops` by reading chunks, or raise UnsupportedDataset."""
    # We write into slices of the output array as raw bytes.
    if not out.flags.c_contiguous:
        raise UnsupportedDataset("the output array is not C-contiguous")
    if out.dtype.hasobject:
        raise UnsupportedDataset(f"dtype {out.dtype} is not a plain number")

    with Planner(out.dtype, roi) as planner:
        jobs = planner.plan(ops)

        if threads is None:
            threads = default_num_threads(
                THREADS_UNFILTERED if planner.all_unfiltered()
                else THREADS_FILTERED
            )

        read_debug("reading %.2f MB as %d jobs on %d threads",
               out.nbytes / 1e6, len(jobs), min(threads, len(jobs)))
        run(jobs, out, threads)


def read(out, ops, roi=(), parallel=-1):
    """Fill `out` from `ops`, taking `roi` of each entry.

    `parallel` is how many threads to read the chunks with. 0 goes through h5py
    on this thread instead. -1 reads the chunks directly if these datasets allow
    it, and through h5py if they don't. A positive number insists on reading the
    chunks directly, raising :exc:`UnsupportedDataset` if that isn't possible.

    Returns True if we read the chunks ourselves, False if h5py did. Set
    ``EXTRA_DATA_DIRECT_READ_DEBUG=1`` to have each read report which it was.
    """
    if len(ops) == 0:
        return True

    if parallel != 0:
        try:
            read_directly(out, ops, roi, parallel if parallel > 0 else None)
            return True
        except UnsupportedDataset as e:
            if parallel > 0:
                raise

            read_debug("h5py: %s", e)

    for op in ops:
        op.read_into(out[op.dest_slice], roi)

    return False
