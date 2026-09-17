# coding: utf-8
"""Expose data to different interface

ZMQStream explose to a ZeroMQ socket in a REQ/REP pattern.

Copyright (c) 2017, European X-Ray Free-Electron Laser Facility GmbH
All rights reserved.

You should have received a copy of the 3-Clause BSD License along with this
program. If not, see <https://opensource.org/licenses/BSD-3-Clause>
"""

import os
import queue
import threading
import dataclasses
import os.path as osp
import time
from collections import deque
from socket import AF_INET
from functools import partial
from warnings import warn
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor, Future

import numpy as np
import xarray as xr
from karabo_bridge import ServerInThread
from karabo_bridge.server import Sender
from psutil import net_if_addrs

from . import direct_read
from .components import XtdfDetectorBase, MultimodKeyData, XtdfImageMultimodKeyData
from .exceptions import SourceNameError
from .reader import RunDirectory, H5File, by_id, by_index
from .stacking import stack_detector_data
from .keydata import KeyData
from .utils import default_num_threads


__all__ = ['ZMQStreamer', 'serve_files', 'MemoryStreamer']


def find_infiniband_ip():
    """Find the first infiniband IP address

    :returns: str
        IP of the first infiniband interface if it exists else '*'
    """
    addrs = net_if_addrs()
    for addr in addrs.get('ib0', ()):
        if addr.family == AF_INET:
            return addr.address
    return '*'


class ZMQStreamer(ServerInThread):
    def __init__(self, port, sock='REP', maxlen=10, protocol_version='2.2',
                 dummy_timestamps=False):
        warn("Please use :ref:karabo_bridge.ServerInThread instead",
             DeprecationWarning, stacklevel=2)

        endpoint = f'tcp://*:{port}'
        super().__init__(endpoint, sock=sock, maxlen=maxlen,
                         protocol_version=protocol_version,
                         dummy_timestamps=dummy_timestamps)

@dataclasses.dataclass
class PartialDataArray:
    name: str
    attrs: dict
    dims: tuple
    coords: dict
    buffer_slot: int
    shape: tuple

class SlotBuffer:
    """Memory divided into a fixed number of equally sized slots.

    A reader thread fills a slot, the consumer reads it out, and the slot is
    then free to be used again. Fixing the number of slots is what bounds how
    much memory is needed however much data is streamed: a longer stream just
    re-uses slots more often.

    `allocator` may be given to allocate the memory some other way; it's called
    with the shape and dtype name of the whole buffer, and must return the name
    of the buffer and something supporting the buffer protocol. This is what
    lets a caller in another language hand us memory it owns.
    """
    def __init__(self, n_slots, slot_shape, dtype, allocator=None):
        self.n_slots = n_slots
        self.slot_shape = tuple(slot_shape)
        self.dtype = np.dtype(dtype)
        self.shape = (n_slots,) + self.slot_shape
        self.slot_bytes = (self.dtype.itemsize
                           * int(np.prod(self.slot_shape, dtype=np.intp)))

        if allocator is None:
            self.name = None
            self._array = np.empty(self.shape, dtype=self.dtype)
        else:
            # The caller's allocator owns the memory, so we must not free it
            self.name, buffer = allocator(self.shape, self.dtype.name)
            self._array = np.ndarray(self.shape, dtype=self.dtype, buffer=buffer)

    def __getitem__(self, slot):
        return self._array[slot]

    def view(self, slot, shape):
        """View the start of a slot's memory as a contiguous array of `shape`."""
        n = int(np.prod(shape, dtype=np.intp))
        return self._array[slot].reshape(-1)[:n].reshape(shape)

    def release(self):
        """Let go of the buffer, so its memory can be reclaimed.

        This only drops our own reference: a consumer still holding a slot keeps
        the memory alive, and goes on reading valid data. With an `allocator`
        the memory belongs to the caller and isn't ours to free at all.
        """
        self._array = None


class SlotStream:
    """Keep the slots of a buffer busy with jobs, and give back filled slots.

    Each job is submitted to the reader threads along with a free slot to write
    into, so at most `n_slots` of them are in flight at once. Iterating gives
    back ``(job, slot, result)`` in the order the jobs were given, and the
    consumer calls :meth:`release` once it's finished with a slot's contents.

    Submitting has to wait for a free slot, so it's done on a thread of its own:
    that lets a single-threaded consumer drive the whole thing by iterating,
    without deadlocking against its own unreleased slots.

    `func` is called in a reader thread as ``func(job, buffer, slot)``. `stop`
    may be a callable which returns True to abandon the stream, checked while
    waiting for a slot to come free.
    """
    def __init__(self, buffer, jobs, func, pool, stop=None):
        self.buffer = buffer
        self._func = func
        self._pool = pool
        self._stop = stop
        self._closed = False

        # Every slot is free to begin with
        self._free_slots = queue.Queue()
        for slot in range(buffer.n_slots):
            self._free_slots.put(slot)

        # Results are taken from this in the order the jobs were submitted
        self._submitted = queue.Queue()
        self._feeder_error = None
        self._feeder = threading.Thread(target=self._feed, args=(iter(jobs),),
                                        daemon=True)
        self._feeder.start()

    def _acquire(self):
        """Wait for a free slot, returning None if the stream is stopping."""
        while True:
            # The timeout lets us periodically check whether we should give up.
            try:
                return self._free_slots.get(timeout=0.1)
            except queue.Empty:
                if self._closed or (self._stop is not None and self._stop()):
                    return None

    def release(self, slot):
        """Return a slot to be used again, once it's finished with."""
        self._free_slots.put(slot)

    def _feed(self, jobs):
        try:
            for job in jobs:
                slot = self._acquire()
                if slot is None:
                    return

                future = self._pool.submit(self._func, job, self.buffer, slot)
                self._submitted.put((job, slot, future))
        except BaseException as e:
            self._feeder_error = e
        finally:
            # Mark the end of the stream from a finally block, so that a
            # consumer is never left waiting for results that can't come.
            self._submitted.put(None)

    def __iter__(self):
        while (item := self._submitted.get()) is not None:
            job, slot, future = item
            # Re-raise anything the reader hit
            yield job, slot, future.result()

        if self._feeder_error is not None:
            raise self._feeder_error

    def close(self):
        """Stop submitting jobs and wait for those already running.

        The buffer isn't freed here: a consumer may still be holding slots of
        it, and only the buffer's owner knows when that's no longer true.
        """
        self._closed = True
        self._feeder.join()

        # Cancel what hasn't started, then wait for what has. Nothing more can
        # be submitted now that the feeder has finished, so an empty queue means
        # we've seen everything: we can't wait for the end-of-stream marker,
        # which iterating may already have taken.
        pending = []
        while True:
            try:
                item = self._submitted.get_nowait()
            except queue.Empty:
                break

            if item is not None and not item[2].cancel():
                pending.append(item[2])

        futures.wait(pending)


def generate_kd_data(kd, out=None):
    if out is None:
        out = np.zeros(kd.buffer_shape(), dtype=kd.dtype)

    # Mimics kd.xarray(), so the train axis is one entry per train ID here and
    # read_kd() splits it up the same way for mock and real reads alike.
    dims = ["trainId", *[f"dim_{i}" for i in range(1, kd.ndim)]]
    return xr.DataArray(out, dims=dims,
                        coords=dict(trainId=kd.train_id_coordinates()))

def train_axis(obj):
    # Multi-module data has a module axis in front
    if isinstance(obj, MultimodKeyData):
        return 1
    else:
        return 0

def read_1_thread(read):
    """Read on this thread, taking the HDF5 lock only if we have to.

    Reading the chunks ourselves leaves the lock alone, so the reader threads
    run at once; data we can't read that way (strings, missing chunks) still
    has to go through HDF5. Either way it's one thread: the reader threads
    already give us as much parallelism as we want, and each starting its own
    pool would only make them fight over the same cores.
    """
    try:
        return read(1)
    except direct_read.UnsupportedDataset:
        return read(0)


def read_kd(kd, buffer, slot, xarray, mock_io):
    """Read one chunk of a source into a slot. Runs in a reader thread."""
    n_trains = len(kd.train_ids)
    out = buffer.view(slot, kd.buffer_shape())

    # Every train holds the same number of entries, so the train axis can be
    # split into one axis for trains and one for each train's entries.
    t = train_axis(kd)
    shape = out.shape[:t] + (n_trains, out.shape[t] // n_trains) + out.shape[t + 1:]

    if xarray and isinstance(kd, MultimodKeyData):
        # kd.xarray() would read the pulse IDs for its labels, which we'd only
        # throw away, so the labels are made here instead.
        if not mock_io:
            read_1_thread(lambda p: kd.ndarray(out=out, parallel=p))
        # XTDF image data has an entry per pulse. These are labelled with
        # indices rather than pulse IDs, so the axis has no coordinate.
        if isinstance(kd, XtdfImageMultimodKeyData):
            entry_dim = "pulseIndex"
        else:
            entry_dim = "entry"

        dims = (*kd.dimensions[:t], "trainId", entry_dim, *kd.dimensions[t + 1:])
        coords = {"module": kd.modules, "trainId": np.asarray(kd.train_ids)}
        return PartialDataArray(None, {}, dims, coords, slot, shape)
    elif xarray:
        x = (generate_kd_data(kd, out=out) if mock_io else
             read_1_thread(lambda p: kd.xarray(out=out, parallel=p)))
        # Only the labels are sent back; the data itself stays in the buffer.
        # kd.xarray() labels one entry per train ID, which we split up into a
        # train axis and an axis for each train's entries.
        return PartialDataArray(x.name, x.attrs,
                                (x.dims[0], "entry", *x.dims[1:]),
                                {"trainId": np.asarray(kd.train_ids)},
                                slot, shape)
    else:
        if not mock_io:
            read_1_thread(lambda p: kd.ndarray(out=out, parallel=p))
        return PartialDataArray(None, None, None, None, slot, shape)

def select_chunks(kd, tid_chunks):
    # The chunk index goes along with each job so that reads can be timed
    for i, tids in enumerate(tid_chunks):
        yield i, kd.select_trains(by_id[tids])

def align_sources(source_objects):
    if len(source_objects) == 0:
        raise ValueError("At least one source is required to stream")

    common_tids = None
    for obj in source_objects.values():
        if isinstance(obj, MultimodKeyData):
            # min_modules takes care of dropping trains for detector components
            tids = obj.train_ids
        else:
            tids = obj.drop_empty_trains().train_ids

        # train_ids is a list, and np.intersect1d() only makes an array of it
        # when there are at least two sources.
        tids = np.asarray(tids)

        if common_tids is None:
            common_tids = tids
        else:
            common_tids = np.intersect1d(common_tids, tids)

    return common_tids

@dataclasses.dataclass
class EventIterator:
    """Consume a stream by waiting on a file descriptor.

    This is for consumers that have their own event loop (e.g. Julia) and can't
    afford to block a thread inside `queue.get()`. `fd` is an eventfd that
    becomes readable when parts are ready; it's only a wake-up call, so after
    waiting on it call `get()` until it returns None, then wait again. `ended`
    says whether the stream is finished. Call `done()` with a part when finished
    with it so its slots can be re-used.
    """

    streamer: "MemoryStreamer"
    fd: int
    stream_future: Future
    ended: bool = False

    def get(self):
        """Retrieve one ready part, or None if none is ready. Never blocks.

        Once the stream has ended `ended` is set and this keeps returning None,
        so a consumer that polls again after the end gets the same answer rather
        than an error.
        """
        if self.ended:
            return None

        if self.fd < 0:
            raise ValueError("This EventIterator has already been closed")

        # Clear the wake-up before checking the queue, so a part queued in
        # between can't be missed: its wake-up comes after the clear.
        try:
            os.eventfd_read(self.fd)
        except BlockingIOError:
            pass

        try:
            td = self.streamer._train_ready_queue.get_nowait()
        except queue.Empty:
            return None

        if td is None:
            # Set this before re-raising below, so that the end of the stream is
            # recorded whether or not the producer died.
            self.ended = True

            # Re-raise whatever killed the producer, if anything did
            self.stream_future.result()
        else:
            td._handout_time = time.perf_counter()

        return td

    def done(self, train_data):
        """Return a part's slots to the streamer to be re-used."""
        self.streamer._return_slots(train_data)

    def close(self):
        """Stop the stream and release the fd. Safe to call more than once."""
        if self.fd < 0:
            return

        self.streamer.cancel()
        try:
            self.stream_future.result()
        finally:
            # Note that we can't ask the kernel whether the fd is still ours to
            # close, because fd numbers are reused: a check would happily report
            # a *different* file as open and we would close that instead. Hence
            # tracking it ourselves, by using a negative fd to mean 'closed'.
            fd, self.fd = self.fd, -1
            os.close(fd)

class MemoryStreamer:
    def __init__(self, pool_size=None):
        # Reads go straight into this process's own memory, so the readers are
        # threads: the work that matters (pread and decompression) doesn't hold
        # the GIL, and there's nothing to copy back. pool_size is how many
        # chunks are read at once; each read is single-threaded itself.
        self._pool = ThreadPoolExecutor(
            max_workers=pool_size or default_num_threads(),
            thread_name_prefix='extra-data-reader',
        )
        self._stream_pool = ThreadPoolExecutor(max_workers=1)

        self._cache = dict()
        self._ready_fd = None
        self._buffers = []
        self._reset_timings()

    def _release_buffers(self):
        """Let go of the buffers from the last stream.

        Parts a consumer is still holding keep their buffer alive, so this only
        drops our own reference. It's still done when the *next* stream starts
        rather than when one finishes, so that a consumer working through the
        tail of a stream isn't racing us for the memory.
        """
        for buffer in self._buffers:
            buffer.release()

        self._buffers = []

    def close(self):
        """Stop streaming, shut the reader threads down and drop the buffers."""
        self.cancel()
        self._stream_pool.shutdown()
        self._pool.shutdown()
        self._release_buffers()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _start_stream(self, source_objects, *args, use_eventfd=False, **kwargs):
        for obj in source_objects.values():
            if not isinstance(obj, (KeyData, MultimodKeyData)):
                raise TypeError(f"Unsupported object: {obj}")

        self._train_ready_queue = queue.Queue(0)
        self._slot_streams = { }
        self._slot_refcounts = { }
        self._slot_refcount_lock = threading.Lock()
        self._stop_streaming = False
        self._buffers_by_key = { }
        self._common_tids = align_sources(source_objects)
        self._reset_timings()

        # Freeing the previous stream's segments here (rather than at the end
        # of _stream) means parts stay valid until the next stream is started.
        self._release_buffers()

        if use_eventfd:
            # A wake-up call only, the queue is what says whether a part is
            # ready. EFD_NONBLOCK lets get() clear it without hanging when
            # there's nothing to clear.
            self._ready_fd = os.eventfd(0, os.EFD_NONBLOCK)
        else:
            # iterate() blocks on the queue directly and needs no fd
            self._ready_fd = None

        # It's a bit odd to use a pool to submit single jobs, but this is the
        # easiest way of getting any exception thrown to bubble up.
        return self._stream_pool.submit(self._stream_guarded, source_objects,
                                        *args, **kwargs)

    def _reset_timings(self):
        self._t0 = time.perf_counter()
        self._events = []

    def _record(self, kind, start, end, key=None, chunk=None, train_id=None):
        # list.append() is atomic, so the reader threads can record without a lock
        self._events.append((kind, key, chunk, train_id,
                             threading.current_thread().name,
                             start - self._t0, end - self._t0))

    def timings(self):
        """Get the timings of the last stream as a DataFrame.

        Each row is one span of time, with `start` and `end` in seconds since
        the stream started. `kind` is one of:

        - ``read``: a reader thread reading one chunk of the source `key`.
        - ``wait``: ``iterate()`` blocking because no part was ready yet. Lots
          of this means the IO is slower than the consumer.
        - ``compute``: the consumer holding a part, from when it was handed out
          until it was given back.

        Gaps in the reads while the consumer computes mean the consumer is
        slower than the IO.
        """
        import pandas as pd

        columns = ["kind", "key", "chunk", "train_id", "thread", "start", "end"]
        df = pd.DataFrame(list(self._events), columns=columns)
        df["duration"] = df["end"] - df["start"]
        return df.sort_values("start", ignore_index=True)

    def plot_timings(self, ax=None, merge_gap=0.01, collapse_readers=True):
        """Plot the timings of the last stream as a timeline, one row per thread.

        This shows how long IO and compute take, and how much they overlap.
        See `timings()` for what each kind of span means.

        Spans of the same kind in a row that are less than `merge_gap` seconds
        apart are drawn as one. Pass 0 to draw every span separately.

        With `collapse_readers` the reader threads share a single row, showing
        when any of them was reading. This is easier to compare with a serial
        consumer than a row per thread.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        df = self.timings()
        if collapse_readers:
            # Overlapping reads are then merged into one span below
            is_reader = df["thread"].str.startswith("extra-data-reader")
            df.loc[is_reader, "thread"] = "extra-data-readers"

        if ax is None:
            _, ax = plt.subplots(figsize=(12, 4))

        # Reader threads first in numerical order, then the consumer's
        def thread_order(name):
            return not name.startswith("extra-data-reader"), len(name), name

        threads = sorted(df["thread"].unique(), key=thread_order)
        colors = {"read": "tab:blue", "wait": "tab:red", "compute": "tab:green"}

        bar_height = 0.8
        for thread, spans in df.groupby("thread"):
            # Only spans that directly follow each other are merged, otherwise
            # e.g. waits a few ms apart would swallow the compute between them.
            # The spans are sorted by start time.
            merged = []
            for kind, start, end in zip(spans["kind"], spans["start"], spans["end"]):
                if len(merged) > 0 and merged[-1][0] == kind and start - merged[-1][2] < merge_gap:
                    merged[-1][2] = max(merged[-1][2], end)
                else:
                    merged.append([kind, start, end])

            i = threads.index(thread)
            for kind, color in colors.items():
                bars = [(start, end - start) for k, start, end in merged if k == kind]
                # The edges keep back-to-back spans apart
                ax.broken_barh(bars, (i - bar_height / 2, bar_height), color=color,
                               edgecolor="white", linewidth=0.5)

        ax.set_yticks(np.arange(len(threads)),
                      labels=[name.removeprefix("extra-data-") for name in threads])
        ax.invert_yaxis()
        ax.set_xlabel("Time since stream start [s]")
        ax.legend(handles=[Patch(color=c, label=k) for k, c in colors.items()],
                  loc="lower center", bbox_to_anchor=(0.5, 1), ncol=len(colors), frameon=False)
        ax.figure.tight_layout()

        return ax

    def _read_chunk(self, key, job, buffer, slot, **kwargs):
        chunk, kd = job
        start = time.perf_counter()
        result = read_kd(kd, buffer, slot, **kwargs)
        self._record("read", start, time.perf_counter(), key=key, chunk=chunk)
        return result

    def _return_slots(self, td):
        """Tell the submitter threads that a part's slots can be re-used."""
        self._record("compute", td._handout_time, time.perf_counter(), chunk=td._chunk,
                     train_id=int(np.ravel(td.trainId)[0]))

        for key, slot in td._key_slots.items():
            # A slot holds a whole chunk, so it's only free once every part of
            # that chunk has been returned.
            with self._slot_refcount_lock:
                self._slot_refcounts[key, td._chunk] -= 1
                free = self._slot_refcounts[key, td._chunk] == 0
                if free:
                    del self._slot_refcounts[key, td._chunk]

            if free:
                self._slot_streams[key].release(slot)

    def _put_ready(self, td):
        # Always put before notifying, so that a token on the eventfd is a
        # guarantee to the consumer that get_nowait() will succeed.
        self._train_ready_queue.put(td)
        if self._ready_fd is not None:
            os.eventfd_write(self._ready_fd, 1)

    def _stream_guarded(self, source_objects, *args, **kwargs):
        try:
            self._stream(source_objects, *args, **kwargs)
        finally:
            # Stop submitting new chunks and wait for the reads still in
            # progress, so that no worker is writing into a buffer once we're
            # done with it. Leaving them running would also let them carry on
            # into the next stream, which resets _stop_streaming.
            for stream in self._slot_streams.values():
                stream.close()

            # Push the end-of-stream sentinel from a finally block so that the
            # consumer is woken up even if _stream() raised. Consumers waiting
            # on the eventfd have no timeout to fall back on, so without this
            # they would hang forever.
            self._put_ready(None)

    def event_iterator(self, *args, **kwargs):
        """Stream to a consumer with its own event loop, see `EventIterator`.

        Note that no fd is needed for the consumer to return slots, because
        a stream's free slots are unbounded and so releasing never blocks.
        """
        stream_future = self._start_stream(*args, use_eventfd=True, **kwargs)
        return EventIterator(self, self._ready_fd, stream_future)

    def cancel(self):
        self._stop_streaming = True

    def iterate(self, *args, **kwargs):
        future = self._start_stream(*args, **kwargs)

        try:
            # _stream_guarded() always pushes the end-of-stream sentinel, even
            # if the producer died, so we can block here without a timeout.
            while True:
                # Only time it when there's actually a wait
                try:
                    td = self._train_ready_queue.get_nowait()
                except queue.Empty:
                    start = time.perf_counter()
                    td = self._train_ready_queue.get()
                    self._record("wait", start, time.perf_counter())

                if td is None:
                    break

                td._handout_time = time.perf_counter()
                yield td
                self._return_slots(td)
        finally:
            self.cancel()
            future.result()

    def _stream(self, source_objects, *,
                trains_per_part=1, reader_trains_per_chunk=5, n_buffer_slots=15,
                autosqueeze=True, xarray=True, allocator=None, mock_io=False):
        """
        Design overview:
        - We accept a dictionary of names to 'source objects', which currently
          may be a KeyData or DataArray object.
        - Slow data sources will be loaded immediately. Fast data sources will
          be loaded in chunks in parallel by a pool of reader threads.
        - The loading of fast data chunks is asynchronous. Each source gets its
          own buffer with space for multiple chunks (these are called 'slots'
          in the buffer), a queue to indicate when a slot is free to be reused,
          and a thread that keeps submitting new chunks of the source to load.
        - The streamer allows an arbitrary number of slots to be in use by a
          consumer at once to allow for parallel processing. The consumer is
          responsible for returning each part when it's finished with it, and a
          slot is re-used once every part of its chunk has been returned.
        - Streaming can be stopped by setting the `self._stop_streaming` flag.

        Terminology:
        - Part: The thing that gets passed to the user, in units of trains.
        - Chunk: The thing that gets read from disk in one go, in units of
          parts. For efficiency this is often the size of multiple parts. Note
          that chunks at the end of a run may be smaller than others.
        - Buffer: A single array that the source data will be read into.
          Currently there is one buffer per fast data source.
        - Slot: An element within a source buffer. A buffer contains multiple
          slots that can be used concurrently. There can be an arbitrary
          number of slots per source. A slot is the size of 1 chunk.
        """

        # Figure out how big the reader chunks should be
        if reader_trains_per_chunk is None:
            max_trains = 10 // trains_per_part
            if max_trains == 0:
                reader_trains_per_chunk = trains_per_part
            else:
                reader_trains_per_chunk = trains_per_part * max_trains

        if reader_trains_per_chunk % trains_per_part != 0:
            raise ValueError("reader_trains_per_chunk must be evenly divisible by trains_per_part")

        # Note that the buffers are held by the streamer rather than by this
        # function, because consumers may still be working through arrays backed
        # by them after _stream() has returned.

        # Note that we make a new dict rather than modifying the callers
        source_objects = { key: obj.select_trains(by_id[self._common_tids])
                           for key, obj in source_objects.items() }

        # Split the sources into fast data, which is read in chunks by the
        # reader threads, and slow data, which is read up-front.
        fast_sources = { }
        slow_sources = { }
        for key, obj in source_objects.items():
            if obj.is_instrument:
                fast_sources[key] = obj
            elif obj.is_control:
                slow_sources[key] = obj
            else:
                raise ValueError(f"Cannot stream {obj.source}, only CONTROL and "
                                 f"INSTRUMENT sources are supported (not {obj.section})")

        # Load all the slow data. The cache is keyed on what the data actually
        # is, because the user's label for a source says nothing about that: the
        # same label may well refer to different data in a later stream.
        cache_keys = { }
        for key, obj in slow_sources.items():
            cache_key = (obj.source, obj.key, mock_io, self._common_tids.tobytes())
            cache_keys[key] = cache_key
            if cache_key not in self._cache:
                self._cache[cache_key] = generate_kd_data(obj) if mock_io else obj.xarray()

        # Split the train IDs we have into chunks, and each chunk into parts
        tid_chunks = np.split(self._common_tids,
                              np.arange(reader_trains_per_chunk, len(self._common_tids), reader_trains_per_chunk))
        tid_parts_per_chunk = [ ]
        for tids in tid_chunks:
            tid_parts = np.array_split(tids, max(1, len(tids) // trains_per_part))
            tid_parts_per_chunk.append([x for x in tid_parts if len(x) > 0])

        # Create a custom dataclass using the names from the user
        internal_fields = ["trainId", "idx", "_chunk", "_key_slots", "_handout_time"]
        reserved = set(internal_fields) & set(source_objects)
        if reserved:
            raise ValueError(f"These names are reserved and can't be used for sources: {sorted(reserved)}")
        TrainData = dataclasses.make_dataclass("TrainData",
                                               internal_fields + list(source_objects.keys()))

        # An instrument source may hold more than one entry (e.g. a detector
        # frame) per train, which get an axis of their own. A slot is a
        # rectangle, so a source whose trains don't all hold the same number of
        # entries can't be streamed.
        entries_per_train = { }
        for key, kd in fast_sources.items():
            counts = np.unique(kd.data_counts(labelled=False))
            if len(counts) > 1:
                if isinstance(kd, MultimodKeyData):
                    source = kd.det.detector_name
                else:
                    source = kd.source

                raise ValueError(f"Cannot stream {source}/{kd.key}, its trains "
                                 f"hold different numbers of entries: {counts}")

            entries_per_train[key] = int(counts.max(initial=1))

        train_axes = { key: train_axis(kd) for key, kd in fast_sources.items() }

        # Allocate a buffer and a free-slot queue for each fast data source
        for key, kd in fast_sources.items():
            # n_buffer_slots may be a dict to give each source its own number of
            # slots, which is useful when their sizes differ a lot.
            if isinstance(n_buffer_slots, dict):
                source_slots = n_buffer_slots[key]
            else:
                source_slots = n_buffer_slots

            # A slot holds one chunk, i.e. reader_trains_per_chunk trains
            slot_shape = kd.select_trains(by_index[:reader_trains_per_chunk]).buffer_shape()
            buffer = SlotBuffer(source_slots, slot_shape, kd.dtype,
                                allocator=allocator)
            self._buffers_by_key[key] = buffer
            self._buffers.append(buffer)

        # Give each fast source a stream, which keeps that source's slots busy
        # by submitting a job for each chunk as slots come free. Results come
        # back in chunk order, so the loop below can just take the next one from
        # each source.
        for key, kd in fast_sources.items():
            # Selecting is slow for multi-module data, so each chunk is only
            # selected when the stream is ready to read it.
            chunk_kds = select_chunks(kd, tid_chunks)
            read_chunk = partial(self._read_chunk, key, xarray=xarray, mock_io=mock_io)
            self._slot_streams[key] = SlotStream(
                self._buffers_by_key[key], chunk_kds, read_chunk,
                pool=self._pool, stop=lambda: self._stop_streaming,
            )

        source_results = { key: iter(stream)
                           for key, stream in self._slot_streams.items() }

        # Iterate through the chunks to give the data to the users loop
        for i, read_tids in enumerate(tid_chunks):
            # Quit if requested
            if self._stop_streaming:
                break

            # Take this chunk from each source, waiting for the data to be
            # loaded. A stream that has been stopped runs out early, which is
            # the other way this loop ends.
            partial_results = { }
            for key, results in source_results.items():
                chunk_result = next(results, None)
                if chunk_result is None:
                    break

                partial_results[key] = chunk_result[2]

            if len(partial_results) < len(fast_sources):
                break

            # Re-build the DataArrays
            results = { }
            for key, partial_array in partial_results.items():
                array = self._buffers_by_key[key].view(partial_array.buffer_slot, partial_array.shape)
                if xarray:
                    results[key] = xr.DataArray(array,
                                                name=partial_array.name,
                                                dims=partial_array.dims,
                                                coords=partial_array.coords,
                                                attrs=partial_array.attrs)
                else:
                    results[key] = array

            # Split the chunk into parts of the size the user requested
            tid_parts = tid_parts_per_chunk[i]

            # Tell _return_slots() how many parts it should expect back before
            # this chunk's slots are free again. This must happen before any of
            # the parts are handed out.
            with self._slot_refcount_lock:
                for key in partial_results:
                    self._slot_refcounts[key, i] = len(tid_parts)

            for iter_tids in tid_parts:
                start_idx = np.searchsorted(read_tids, iter_tids[0])
                end_idx = start_idx + len(iter_tids)

                # If the user wants autosqueezing and they're iterating one
                # train at a time, use a scalar train ID to select instead of a
                # list of 1 train ID. This will make xarray drop the trainId
                # dimension.
                squeeze = autosqueeze and trains_per_part == 1 and len(iter_tids) == 1
                if squeeze:
                    iter_tids = iter_tids[0]

                # A source holding a single entry per train has no need of an
                # axis for its entries, so that one is indexed away.
                iter_results = { }
                for key, array in results.items():
                    trains = start_idx if squeeze else np.s_[start_idx:end_idx]
                    entries = 0 if entries_per_train[key] == 1 else np.s_[:]
                    leading = (np.s_[:],) * train_axes[key]
                    iter_results[key] = array[(*leading, trains, entries)]

                # Merge slow data
                for key in slow_sources:
                    iter_results[key] = self._cache[cache_keys[key]].sel(trainId=iter_tids)

                iter_results["trainId"] = iter_tids
                iter_results["idx"] = np.searchsorted(self._common_tids, iter_tids)
                iter_results["_chunk"] = i
                iter_results["_key_slots"] = { k: partial_array.buffer_slot
                                               for k, partial_array in partial_results.items() }
                # Set when the part is given to the consumer, to time its compute
                iter_results["_handout_time"] = None
                td = TrainData(**iter_results)
                self._put_ready(td)

        # Note that the streams are closed and the end-of-stream sentinel is
        # pushed by _stream_guarded(), so that it happens even if we raise.

        # Note that the buffers are not released here, see _release_buffers()
        # for why.

def _iter_trains(data, merge_detector=False):
    """Iterate over trains in data and merge detector tiles in a single source

    :data: DataCollection
    :merge_detector: bool
        if True and data contains detector data (e.g. AGIPD) individual sources
        for each detector tiles are merged in a single source. The new source
        name keep the original prefix, but replace the last 2 part with
        '/DET/APPEND'. Individual sources are removed from the train data

    :yield: dict
        train data
    """
    det, source_name = None, ''
    if merge_detector:
        for detector in XtdfDetectorBase.__subclasses__():
            try:
                det = detector(data)
                source_name = f'{det.detector_name}/DET/APPEND'
            except SourceNameError:
                continue
            else:
                break

    for tid, train_data in data.trains():
        if not train_data:
            continue

        if det is not None:
            det_data = {
                k: v for k, v in train_data.items()
                if k in det.data.detector_sources
            }

            # get one of the module to reference other datasets
            train_data[source_name] = mod_data = next(iter(det_data.values()))

            stacked = stack_detector_data(det_data, 'image.data')
            mod_data['image.data'] = stacked
            mod_data['metadata']['source'] = source_name

            if 'image.gain' in mod_data:
                stacked = stack_detector_data(det_data, 'image.gain')
                mod_data['image.gain'] = stacked
            if 'image.mask' in mod_data:
                stacked = stack_detector_data(det_data, 'image.mask')
                mod_data['image.mask'] = stacked

            # remove individual module sources
            for src in det.data.detector_sources:
                del train_data[src]

        yield tid, train_data


def _drop_object_arrays(data):
    """Drop arrays of Python objects (e.g. strings) from the data.

    E.g. availableScenes, interfaces properties. These break serialisation and
    are unlikely to be useful.
    """
    for d in data.values():
        to_delete = []
        for k, v in d.items():
            if isinstance(v, np.ndarray) and v.dtype == np.dtype(object):
                to_delete.append(k)
                if k.endswith('.value') and (tsk := k[:-6] + '.timestamp') in d:
                    to_delete.append(tsk)

        for k in to_delete:
            del d[k]


def serve_files(path, port, source_glob='*', key_glob='*', **kwargs):
    """Stream data from files through a TCP socket.

    Parameters
    ----------
    path: str
        Path to the HDF5 file or file folder.
    port: str or int
        A ZMQ endpoint (e.g. 'tcp://*:44444') or a TCP port to bind the socket
        to. Integers or strings of all digits are treated as port numbers.
    source_glob: str
        Only stream sources matching this glob pattern.
        Streaming data selectively is more efficient than streaming everything.
    key_glob: str
        Only stream keys matching this glob pattern in the selected sources.
    append_detector_modules: bool
        Combine multi-module detector data in a single data source (sources for
        individual modules are removed). The last section of the source name is
        replaces with 'APPEND', example:
            'SPB_DET_AGIPD1M-1/DET/#CH0:xtdf' -> 'SPB_DET_AGIPD1M-1/DET/APPEND'

        Supported detectors: AGIPD, DSSC, LPD
    dummy_timestamps: bool
        Whether to add mock timestamps if the metadata lacks them.
    use_infiniband: bool
        Use infiniband interface if available (if port specifies a TCP port)
    sock: str
        socket type - supported: REP, PUB, PUSH (default REP).
    """
    if osp.isdir(path):
        data = RunDirectory(path)
    else:
        data = H5File(path)

    data = data.select(source_glob, key_glob)
    serve_data(data, port, **kwargs)


def serve_data(data, port, append_detector_modules=False,
                dummy_timestamps=False, use_infiniband=False, sock='REP'):
    """Stream data from files through a TCP socket.

    Parameters
    ----------
    data: DataCollection
        The data to be streamed; should already have sources & keys selected.
    port: str or int
        A ZMQ endpoint (e.g. 'tcp://*:44444') or a TCP port to bind the socket
        to. Integers or strings of all digits are treated as port numbers.
    append_detector_modules: bool
        Combine multi-module detector data in a single data source (sources for
        individual modules are removed). The last section of the source name is
        replaces with 'APPEND', example:
            'SPB_DET_AGIPD1M-1/DET/#CH0:xtdf' -> 'SPB_DET_AGIPD1M-1/DET/APPEND'

        Supported detectors: AGIPD, DSSC, LPD
    dummy_timestamps: bool
        Whether to add mock timestamps if the metadata lacks them.
    use_infiniband: bool
        Use infiniband interface if available (if port specifies a TCP port)
    sock: str
        socket type - supported: REP, PUB, PUSH (default REP).
    """
    if isinstance(port, int) or port.isdigit():
        endpt = f'tcp://{find_infiniband_ip() if use_infiniband else "*"}:{port}'
    else:
        endpt = port

    sender = Sender(endpt, sock=sock, dummy_timestamps=dummy_timestamps)
    print(f'Streamer started on: {sender.endpoint}')
    ntrains = len(data.train_ids)

    sent_times = deque([time.monotonic()], 10)
    count = 0
    tid, rate = 0, 0.
    def print_update(end='\r'):
        print(f'Sent {count}/{ntrains} trains - Train ID {tid} - {rate:.1f} Hz', end=end)

    for tid, data in _iter_trains(data, merge_detector=append_detector_modules):
        _drop_object_arrays(data)
        sender.send(data)
        count += 1
        new_time = time.monotonic()
        if count % 5 == 0:
            rate = len(sent_times) / (new_time - sent_times[0])
            print_update()
        sent_times.append(new_time)
    print_update(end='\n')

    # The karabo-bridge code sets linger to 0 so that it doesn't get stuck if
    # the client goes away. But this would also mean that we close the socket
    # when the last messages have been queued but not sent. So if we've
    # successfully queued all the messages, set linger -1 (i.e. infinite) to
    # wait until ZMQ has finished transferring them before the socket is closed.
    sender.server_socket.close(linger=-1)
