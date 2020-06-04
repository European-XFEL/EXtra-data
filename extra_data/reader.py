# coding: utf-8
"""
Collection of classes and functions to help reading HDF5 file generated at
The European XFEL.

Copyright (c) 2017, European X-Ray Free-Electron Laser Facility GmbH
All rights reserved.

You should have received a copy of the 3-Clause BSD License along with this
program. If not, see <https://opensource.org/licenses/BSD-3-Clause>
"""

from collections import defaultdict
from collections.abc import Iterable
import datetime
import fnmatch
import h5py
from itertools import groupby
import logging
from multiprocessing import Pool
import numpy as np
from operator import index
import os
import os.path as osp
import psutil
import re
import signal
import sys
import tempfile
import time
from warnings import warn

from .exceptions import SourceNameError, PropertyNameError, TrainIDError
from .read_machinery import (
    DETECTOR_SOURCE_RE,
    DataChunk,
    FilenameInfo,
    _SliceConstructable,
    _tid_to_slice_ix,
    union_selections,
    contiguous_regions,
    find_proposal,
)
from .run_files_map import RunFilesMap
from . import locality
from .file_access import FileAccess

__all__ = [
    'H5File',
    'RunDirectory',
    'open_run',
    'FileAccess',
    'DataCollection',
    'by_id',
    'by_index',
    'SourceNameError',
    'PropertyNameError',
]

log = logging.getLogger(__name__)

RUN_DATA = 'RUN'
INDEX_DATA = 'INDEX'
METADATA = 'METADATA'


class by_id(_SliceConstructable):
    pass


class by_index(_SliceConstructable):
    pass


class DataCollection:
    """An assemblage of data generated at European XFEL

    Data consists of *sources* which each have *keys*. It is further
    organised by *trains*, which are identified by train IDs.

    You normally get an instance of this class by calling :func:`H5File`
    for a single file or :func:`RunDirectory` for a directory.
    """
    def __init__(self, files, selection=None, train_ids=None, ctx_closes=False):
        self.files = list(files)
        self.ctx_closes = ctx_closes

        # selection: {source: set(keys)}
        # None as value -> all keys for this source
        if selection is None:
            selection = {}
            for f in self.files:
                selection.update(dict.fromkeys(f.control_sources))
                selection.update(dict.fromkeys(f.instrument_sources))
        self.selection = selection

        self.control_sources = set()
        self.instrument_sources = set()
        self._source_index = defaultdict(list)
        for f in self.files:
            self.control_sources.update(f.control_sources.intersection(selection))
            self.instrument_sources.update(f.instrument_sources.intersection(selection))
            for source in (f.control_sources | f.instrument_sources):
                self._source_index[source].append(f)

        # Throw an error if we have conflicting data for the same source
        self._check_source_conflicts()

        self.control_sources = frozenset(self.control_sources)
        self.instrument_sources = frozenset(self.instrument_sources)

        if train_ids is None:
            train_ids = sorted(set().union(*(f.train_ids for f in files)))
        self.train_ids = train_ids

    @staticmethod
    def _open_file(path, cache_info=None):
        try:
            fa = FileAccess(path, _cache_info=cache_info)
        except Exception as e:
            return osp.basename(path), str(e)
        else:
            return osp.basename(path), fa

    @classmethod
    def from_paths(cls, paths, _files_map=None):
        files = []
        uncached = []
        for path in paths:
            cache_info = _files_map and _files_map.get(path)
            if cache_info:
                filename, fa = cls._open_file(path, cache_info=cache_info)
                if isinstance(fa, FileAccess):
                    files.append(fa)
                else:
                    print(f"Skipping file {filename}", file=sys.stderr)
                    print(f"  (error was: {fa})", file=sys.stderr)
            else:
                uncached.append(path)

        if uncached:
            def initializer():
                # prevent child processes from receiving KeyboardInterrupt
                signal.signal(signal.SIGINT, signal.SIG_IGN)

            # cpu_affinity give a list of cpu cores we can use, can be all or a
            # subset of the cores the machine has.
            nproc = min(len(psutil.Process().cpu_affinity()), len(uncached))
            with Pool(processes=nproc, initializer=initializer) as pool:
                for fname, fa in pool.imap_unordered(cls._open_file, uncached):
                    if isinstance(fa, FileAccess):
                        files.append(fa)
                    else:
                        print(f"Skipping file {fname}", file=sys.stderr)
                        print(f"  (error was: {fa})", file=sys.stderr)

        if not files:
            raise Exception("All HDF5 files specified are unusable")

        return cls(files, ctx_closes=True)

    @classmethod
    def from_path(cls, path):
        files = [FileAccess(path)]
        return cls(files, ctx_closes=True)

    def __enter__(self):
        if not self.ctx_closes:
            raise Exception(
                "Only DataCollection objects created by opening "
                "files directly can be used in a 'with' statement, "
                "not those created by selecting from or merging "
                "others."
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Close the files if this collection was created by opening them.
        if self.ctx_closes:
            for file in self.files:
                file.close()

    @property
    def all_sources(self):
        return self.control_sources | self.instrument_sources

    @property
    def detector_sources(self):
        return set(filter(DETECTOR_SOURCE_RE.match, self.instrument_sources))

    def _check_field(self, source, key):
        if source not in self.all_sources:
            raise SourceNameError(source)

        if not self._has_source_key(source, key):
            raise PropertyNameError(key, source)

    def _has_source_key(self, source, key):
        selected_keys = self.selection[source]
        if selected_keys is not None:
            return key in selected_keys

        for f in self._source_index[source]:
            return f.has_source_key(source, key)

    def keys_for_source(self, source):
        """Get a set of key names for the given source

        If you have used :meth:`select` to filter keys, only selected keys
        are returned.

        Only one file is used to find the keys. Within a run, all files should
        have the same keys for a given source, but if you use :meth:`union` to
        combine two runs where the source was configured differently, the
        result can be unpredictable.
        """
        selected_keys = self.selection[source]
        if selected_keys is not None:
            return selected_keys

        # The same source may be in multiple files, but this assumes it has
        # the same keys in all files that it appears in.
        for f in self._source_index[source]:
            return f.get_keys(source)

    # Leave old name in case anything external was using it:
    _keys_for_source = keys_for_source

    def get_entry_shape(self, source, key):
        """Get the shape of a single data entry for the given source & key"""
        self._check_field(source, key)

        for chunk in self._find_data_chunks(source, key):
            # First dimension is number of entries
            return chunk.dataset.shape[1:]

    def get_dtype(self, source, key):
        """Get the numpy data type for the given source & key"""
        self._check_field(source, key)
        for chunk in self._find_data_chunks(source, key):
            return chunk.dataset.dtype

    def _check_data_missing(self, tid) -> bool:
        """Return True if a train does not have data for all sources"""
        for source in self.control_sources:
            file, _ = self._find_data(source, tid)
            if file is None:
                return True

        for source in self.instrument_sources:
            file, pos = self._find_data(source, tid)
            if file is None:
                return True

            groups = {k.partition('.')[0] for k in self.keys_for_source(source)}
            for group in groups:
                _, counts = file.get_index(source, group)
                if counts[pos] == 0:
                    return True

        return False

    def trains(self, devices=None, train_range=None, *, require_all=False):
        """Iterate over all trains in the data and gather all sources.

        ::

            run = Run('/path/to/my/run/r0123')
            for train_id, data in run.select("*/DET/*", "image.data").trains():
                mod0 = data["FXE_DET_LPD1M-1/DET/0CH0:xtdf"]["image.data"]

        Parameters
        ----------

        devices: dict or list, optional
            Filter data by sources and keys.
            Refer to :meth:`select` for how to use this.

        train_range: by_id or by_index object, optional
            Iterate over only selected trains, by train ID or by index.
            Refer to :meth:`select_trains` for how to use this.

        require_all: bool
            False (default) returns any data available for the requested trains.
            True skips trains which don't have all the selected data;
            this only makes sense if you make a selection with *devices*
            or :meth:`select`.

        Yields
        ------

        tid : int
            The train ID of the returned train
        data : dict
            The data for this train, keyed by device name
        """
        dc = self
        if devices is not None:
            dc = dc.select(devices)
        if train_range is not None:
            dc = dc.select_trains(train_range)
        return iter(TrainIterator(dc, require_all=require_all))

    def train_from_id(self, train_id, devices=None):
        """Get train data for specified train ID.

        Parameters
        ----------

        train_id: int
            The train ID
        devices: dict or list, optional
            Filter data by sources and keys.
            Refer to :meth:`select` for how to use this.

        Returns
        -------

        tid : int
            The train ID of the returned train
        data : dict
            The data for this train, keyed by device name

        Raises
        ------
        KeyError
            if `train_id` is not found in the run.
        """
        if train_id not in self.train_ids:
            raise TrainIDError(train_id)

        if devices is not None:
            return self.select(devices).train_from_id(train_id)

        res = {}
        for source in self.control_sources:
            source_data = res[source] = {
                'metadata': {'source': source, 'timestamp.tid': train_id}
            }
            file, pos = self._find_data(source, train_id)
            if file is None:
                continue

            for key in self.keys_for_source(source):
                path = '/CONTROL/{}/{}'.format(source, key.replace('.', '/'))
                source_data[key] = file.file[path][pos]

        for source in self.instrument_sources:
            source_data = res[source] = {
                'metadata': {'source': source, 'timestamp.tid': train_id}
            }
            file, pos = self._find_data(source, train_id)
            if file is None:
                continue

            for key in self.keys_for_source(source):
                group = key.partition('.')[0]
                firsts, counts = file.get_index(source, group)
                first, count = firsts[pos], counts[pos]
                if not count:
                    continue

                path = '/INSTRUMENT/{}/{}'.format(source, key.replace('.', '/'))
                if count == 1:
                    source_data[key] = file.file[path][first]
                else:
                    source_data[key] = file.file[path][first : first + count]

        return train_id, res

    def train_from_index(self, train_index, devices=None):
        """Get train data of the nth train in this data.

        Parameters
        ----------
        train_index: int
            Index of the train in the file.
        devices: dict or list, optional
            Filter data by sources and keys.
            Refer to :meth:`select` for how to use this.

        Returns
        -------

        tid : int
            The train ID of the returned train
        data : dict
            The data for this train, keyed by device name
        """
        train_id = self.train_ids[train_index]
        return self.train_from_id(int(train_id), devices=devices)

    def get_data_counts(self, source, key):
        """Get a count of data points in each train for the given data field.

        Returns a pandas series with an index of train IDs.

        Parameters
        ----------
        source: str
            Source name, e.g. "SPB_DET_AGIPD1M-1/DET/7CH0:xtdf"
        key: str
            Key of parameter within that device, e.g. "image.data".
        """
        import pandas as pd

        self._check_field(source, key)
        seq_series = []

        for f in self._source_index[source]:
            if source in self.control_sources:
                counts = np.ones(len(f.train_ids), dtype=np.uint64)
            else:
                group = key.partition('.')[0]
                _, counts = f.get_index(source, group)
            seq_series.append(pd.Series(counts, index=f.train_ids))

        ser = pd.concat(sorted(seq_series, key=lambda s: s.index[0]))
        # Select out only the train IDs of interest
        train_ids = ser.index.intersection(self.train_ids)
        return ser.loc[train_ids]

    def get_series(self, source, key):
        """Return a pandas Series for a particular data field.

        ::

            s = run.get_series("SA1_XTD2_XGM/XGM/DOOCS", "beamPosition.ixPos")

        This only works for 1-dimensional data.

        Parameters
        ----------

        source: str
            Device name with optional output channel, e.g.
            "SA1_XTD2_XGM/DOOCS/MAIN" or "SPB_DET_AGIPD1M-1/DET/7CH0:xtdf"
        key: str
            Key of parameter within that device, e.g. "beamPosition.iyPos.value"
            or "header.linkId". The data must be 1D in the file.
        """
        import pandas as pd

        self._check_field(source, key)
        name = source + '/' + key
        if name.endswith('.value'):
            name = name[:-6]

        seq_series = []

        if source in self.control_sources:
            data_path = "/CONTROL/{}/{}".format(source, key.replace('.', '/'))
            for f in self._source_index[source]:
                data = f.file[data_path][: len(f.train_ids), ...]
                index = pd.Index(f.train_ids, name='trainId')

                seq_series.append(pd.Series(data, name=name, index=index))

        elif source in self.instrument_sources:
            data_path = "/INSTRUMENT/{}/{}".format(source, key.replace('.', '/'))
            for f in self._source_index[source]:
                group = key.partition('.')[0]
                firsts, counts = f.get_index(source, group)
                trainids = self._expand_trainids(counts, f.train_ids)

                index = pd.Index(trainids, name='trainId')
                data = f.file[data_path][:]
                if not index.is_unique:
                    pulse_id = f.file['/INSTRUMENT/{}/{}/pulseId'.format(source, group)]
                    pulse_id = pulse_id[: len(index), 0]
                    index = pd.MultiIndex.from_arrays(
                        [trainids, pulse_id], names=['trainId', 'pulseId']
                    )
                    # Does pulse-oriented data always have an extra dimension?
                    assert data.shape[1] == 1
                    data = data[:, 0]
                data = data[: len(index)]

                seq_series.append(pd.Series(data, name=name, index=index))
        else:
            raise Exception("Unknown source category")

        ser = pd.concat(sorted(seq_series, key=lambda s: s.index[0]))

        # Select out only the train IDs of interest
        if isinstance(ser.index, pd.MultiIndex):
            train_ids = ser.index.levels[0].intersection(self.train_ids)
            # A numpy array works for selecting, but a pandas index doesn't
            train_ids = np.asarray(train_ids)
        else:
            train_ids = ser.index.intersection(self.train_ids)

        return ser.loc[train_ids]

    def get_dataframe(self, fields=None, *, timestamps=False):
        """Return a pandas dataframe for given data fields.

        ::

            df = run.get_dataframe(fields=[
                ("*_XGM/*", "*.i[xy]Pos"),
                ("*_XGM/*", "*.photonFlux")
            ])

        This links together multiple 1-dimensional datasets as columns in a
        table.

        Parameters
        ----------
        fields : dict or list, optional
            Select data sources and keys to include in the dataframe.
            Selections are defined by lists or dicts as in :meth:`select`.
        timestamps : bool
            If false (the default), exclude the timestamps associated with each
            control data field.
        """
        import pandas as pd

        if fields is not None:
            return self.select(fields).get_dataframe(timestamps=timestamps)

        series = []
        for source in self.all_sources:
            for key in self.keys_for_source(source):
                if (not timestamps) and key.endswith('.timestamp'):
                    continue
                series.append(self.get_series(source, key))

        return pd.concat(series, axis=1)

    def get_array(self, source, key, extra_dims=None, roi=by_index[...]):
        """Return a labelled array for a particular data field.

        ::

            arr = run.get_array("SA3_XTD10_PES/ADC/1:network", "digitizers.channel_4_A.raw.samples")

        This should work for any data.
        The first axis of the returned data will be labelled with the train IDs.

        Parameters
        ----------

        source: str
            Device name with optional output channel, e.g.
            "SA1_XTD2_XGM/DOOCS/MAIN" or "SPB_DET_AGIPD1M-1/DET/7CH0:xtdf"
        key: str
            Key of parameter within that device, e.g. "beamPosition.iyPos.value"
            or "header.linkId".
        extra_dims: list of str
            Name extra dimensions in the array. The first dimension is
            automatically called 'train'. The default for extra dimensions
            is dim_0, dim_1, ...
        roi: by_index
            The region of interest. This expression selects data in all
            dimensions apart from the first (trains) dimension. If the data
            holds a 1D array for each entry, roi=by_index[:8] would get the
            first 8 values from every train. If the data is 2D or more at
            each entry, selection looks like roi=by_index[:8, 5:10] .
        """
        import xarray

        self._check_field(source, key)

        if not isinstance(roi, by_index):
            raise TypeError("roi parameter must be instance of by_index")
        else:
            roi = roi.value
            if not isinstance(roi, tuple):
                roi = (roi,)

        chunks = sorted(
            self._find_data_chunks(source, key),
            key=lambda x: x.train_ids[0] if x.train_ids.size else 0,
        )

        # Figure out the shape of the result array, and the slice for each chunk
        dest_dim0 = 0
        dest_slices = []
        shapes = set()
        dtypes = set()

        for chunk in chunks:
            n = int(np.sum(chunk.counts, dtype=np.uint64))
            dest_slices.append(slice(dest_dim0, dest_dim0 + n))
            dest_dim0 += n
            shapes.add(chunk.dataset.shape[1:])
            dtypes.add(chunk.dataset.dtype)

        if len(shapes) > 1:
            raise Exception("Mismatched data shapes: {}".format(shapes))

        if len(dtypes) > 1:
            raise Exception("Mismatched dtypes: {}".format(dtypes))

        # Find the shape of the array with the ROI applied
        roi_dummy = np.zeros((0,) + shapes.pop()) # extra 0 dim: use less memory
        roi_shape = roi_dummy[np.index_exp[:] + roi].shape[1:]

        chunks_trainids = []
        res = np.empty((dest_dim0,) + roi_shape, dtype=dtypes.pop())

        # Read the data from each chunk into the result array
        for chunk, dest_slice in zip(chunks, dest_slices):
            if dest_slice.start == dest_slice.stop:
                continue

            chunks_trainids.append(
                self._expand_trainids(chunk.counts, chunk.train_ids)
            )

            slices = (chunk.slice,) + roi
            chunk.dataset.read_direct(res[dest_slice], source_sel=slices)

        # Dimension labels
        if extra_dims is None:
            extra_dims = ['dim_%d' % i for i in range(res.ndim - 1)]
        dims = ['trainId'] + extra_dims

        # Train ID index
        coords = {}
        if dest_dim0:
            coords = {'trainId': np.concatenate(chunks_trainids)}

        return xarray.DataArray(res, dims=dims, coords=coords)

    def get_dask_array(self, source, key, labelled=False):
        """Get a Dask array for the specified data field.

        Dask is a system for lazy parallel computation. This method doesn't
        actually load the data, but gives you an array-like object which you
        can operate on. Dask loads the data and calculates results when you ask
        it to, e.g. by calling a ``.compute()`` method.
        See the Dask documentation for more details.

        If your computation depends on reading lots of data, consider creating
        a dask.distributed.Client before calling this.
        If you don't do this, Dask uses threads by default, which is not
        efficient for reading HDF5 files.

        Parameters
        ----------
        source: str
            Source name, e.g. "SPB_DET_AGIPD1M-1/DET/7CH0:xtdf"
        key: str
            Key of parameter within that device, e.g. "image.data".
        labelled: bool
            If True, label the train IDs for the data, returning an
            xarray.DataArray object wrapping a Dask array.
        """
        import dask.array as da
        chunks = sorted(
            self._find_data_chunks(source, key),
            key=lambda x: x.train_ids[0] if x.train_ids.size else 0,
        )

        chunks_darrs = []
        chunks_trainids = []

        for chunk in chunks:
            chunk_dim0 = int(np.sum(chunk.counts))
            chunk_shape = (chunk_dim0,) + chunk.dataset.shape[1:]
            itemsize = chunk.dataset.dtype.itemsize

            # Find chunk size of maximum 2 GB. This is largely arbitrary:
            # we want chunks small enough that each worker can have at least
            # a couple in memory (Maxwell nodes have 256-768 GB in late 2019).
            # But bigger chunks means less overhead.
            # Empirically, making chunks 4 times bigger/smaller didn't seem to
            # affect speed dramatically - but this could depend on many factors.
            # TODO: optional user control of chunking
            limit = 2 * 1024 ** 3
            while np.product(chunk_shape) * itemsize > limit and chunk_dim0 > 1:
                chunk_dim0 //= 2
                chunk_shape = (chunk_dim0,) + chunk.dataset.shape[1:]

            chunks_darrs.append(
                da.from_array(
                    chunk.file.dset_proxy(chunk.dataset_path), chunks=chunk_shape
                )[chunk.slice]
            )
            chunks_trainids.append(
                self._expand_trainids(chunk.counts, chunk.train_ids)
            )

        dask_arr = da.concatenate(chunks_darrs, axis=0)

        if labelled:
            # Dimension labels
            dims = ['trainId'] + ['dim_%d' % i for i in range(dask_arr.ndim - 1)]

            # Train ID index
            coords = {'trainId': np.concatenate(chunks_trainids)}

            import xarray
            return xarray.DataArray(dask_arr, dims=dims, coords=coords)
        else:
            return dask_arr

    def union(self, *others):
        """Join the data in this collection with one or more others.

        This can be used to join multiple sources for the same trains,
        or to extend the same sources with data for further trains.
        The order of the datasets doesn't matter.

        Returns a new :class:`DataCollection` object.
        """
        files = set(self.files)
        train_ids = set(self.train_ids)

        for other in others:
            files.update(other.files)
            train_ids.update(other.train_ids)

        train_ids = sorted(train_ids)
        selection = union_selections([self.selection] +
                                     [o.selection for o in others])

        return DataCollection(files, selection=selection, train_ids=train_ids)

    def _expand_selection(self, selection):
        res = defaultdict(set)
        if isinstance(selection, dict):
            # {source: {key1, key2}}
            # {source: {}} -> all keys for this source
            for source, keys in selection.items():  #
                if source not in self.all_sources:
                    raise SourceNameError(source)

                res[source].update(keys or None)

        elif isinstance(selection, Iterable):
            # selection = [('src_glob', 'key_glob'), ...]
            res = union_selections(
                self._select_glob(src_glob, key_glob)
                for (src_glob, key_glob) in selection
            )
        else:
            raise TypeError("Unknown selection type: {}".format(type(selection)))

        return dict(res)

    def _select_glob(self, source_glob, key_glob):
        source_re = re.compile(fnmatch.translate(source_glob))
        key_re = re.compile(fnmatch.translate(key_glob))
        if key_glob.endswith(('.value', '*')):
            ctrl_key_re = key_re
        else:
            # The translated pattern ends with "\Z" - insert before this
            p = key_re.pattern
            end_ix = p.rindex(r'\Z')
            ctrl_key_re = re.compile(p[:end_ix] + r'(\.value)?' + p[end_ix:])

        matched = {}
        for source in self.all_sources:
            if not source_re.match(source):
                continue

            if key_glob == '*':
                matched[source] = None
            else:
                r = ctrl_key_re if source in self.control_sources else key_re
                keys = set(filter(r.match, self.keys_for_source(source)))
                if keys:
                    matched[source] = keys

        if not matched:
            raise ValueError("No matches for pattern {}"
                             .format((source_glob, key_glob)))
        return matched

    def select(self, seln_or_source_glob, key_glob='*'):
        """Select a subset of sources and keys from this data.

        There are three possible ways to select data:

        1. With two glob patterns (see below) for source and key names::

            # Select data in the image group for any detector sources
            sel = run.select('*/DET/*', 'image.*')

        2. With an iterable of (source, key) glob patterns::

            # Select image.data and image.mask for any detector sources
            sel = run.select([('*/DET/*', 'image.data'), ('*/DET/*', 'image.mask')])

           Data is included if it matches any of the pattern pairs.

        3. With a dict of source names mapped to sets of key names
           (or empty sets to get all keys)::

            # Select image.data from one detector source, and all data from one XGM
            sel = run.select({'SPB_DET_AGIPD1M-1/DET/0CH0:xtdf': {'image.data'},
                              'SA1_XTD2_XGM/XGM/DOOCS': set()})

           Unlike the others, this option *doesn't* allow glob patterns.
           It's a more precise but less convenient option for code that knows
           exactly what sources and keys it needs.

        Returns a new :class:`DataCollection` object for the selected data.

        .. note::
           'Glob' patterns may be familiar from selecting files in a Unix shell.
           ``*`` matches anything, so ``*/DET/*`` selects sources with "/DET/"
           anywhere in the name. There are several kinds of wildcard:

           - ``*``: anything
           - ``?``: any single character
           - ``[xyz]``: one character, "x", "y" or "z"
           - ``[0-9]``: one digit character
           - ``[!xyz]``: one character, *not* x, y or z

           Anything else in the pattern must match exactly. It's case-sensitive,
           so "x" does not match "X".
        """
        if isinstance(seln_or_source_glob, str):
            seln_or_source_glob = [(seln_or_source_glob, key_glob)]
        selection = self._expand_selection(seln_or_source_glob)

        files = [f for f in self.files
                 if f.all_sources.intersection(selection.keys())]

        return DataCollection(files, selection=selection, train_ids=self.train_ids)

    def deselect(self, seln_or_source_glob, key_glob='*'):
        """Select everything except the specified sources and keys.

        This takes the same arguments as :meth:`select`, but the sources and
        keys you specify are dropped from the selection.

        Returns a new :class:`DataCollection` object for the remaining data.
        """

        if isinstance(seln_or_source_glob, str):
            seln_or_source_glob = [(seln_or_source_glob, key_glob)]
        deselection = self._expand_selection(seln_or_source_glob)

        # Subtract deselection from self.selection
        selection = {}
        for source, keys in self.selection.items():
            if source not in deselection:
                selection[source] = keys
                continue

            desel_keys = deselection[source]
            if desel_keys is None:
                continue  # Drop the entire source

            if keys is None:
                keys = self.keys_for_source(source)

            selection[source] = keys - desel_keys

        files = [f for f in self.files
                 if f.all_sources.intersection(selection.keys())]

        return DataCollection(files, selection=selection, train_ids=self.train_ids)

    def select_trains(self, train_range):
        """Select a subset of trains from this data.

        Choose a slice of trains by train ID::

            from extra_data import by_id
            sel = run.select_trains(by_id[142844490:142844495])

        Or select a list of trains::

            sel = run.select_trains(by_id[[142844490, 142844493, 142844494]])

        Or select trains by index within this collection::

            from extra_data import by_index
            sel = run.select_trains(by_index[:5])

        Returns a new :class:`DataCollection` object for the selected trains.

        Raises
        ------
        ValueError
            If given train IDs do not overlap with the trains in this data.
        """
        tr = train_range
        if isinstance(tr, by_id) and isinstance(tr.value, slice):
            # Slice by train IDs
            start_ix = _tid_to_slice_ix(tr.value.start, self.train_ids, stop=False)
            stop_ix = _tid_to_slice_ix(tr.value.stop, self.train_ids, stop=True)
            new_train_ids = self.train_ids[start_ix : stop_ix : tr.value.step]
        elif isinstance(tr, by_index) and isinstance(tr.value, slice):
            # Slice by indexes in this collection
            new_train_ids = self.train_ids[tr.value]
        elif isinstance(tr, by_id) and isinstance(tr.value, (list, np.ndarray)):
            # Select a list of trains by train ID
            new_train_ids = sorted(set(self.train_ids).intersection(tr.value))
            if not new_train_ids:
                raise ValueError(
                    "Given train IDs not found among {} trains in "
                    "collection".format(len(self.train_ids))
                )
        elif isinstance(tr, by_index) and isinstance(tr.value, (list, np.ndarray)):
            # Select a list of trains by index in this collection
            new_train_ids = sorted([self.train_ids[i] for i in tr.value])
        else:
            raise TypeError(type(train_range))

        files = [f for f in self.files
                 if np.intersect1d(f.train_ids, new_train_ids).size > 0]

        return DataCollection(files, selection=self.selection, train_ids=new_train_ids)

    def _check_source_conflicts(self):
        """Check for data with the same source and train ID in different files.
        """
        sources_with_conflicts = set()
        for source, files in self._source_index.items():
            got_tids = np.array([], dtype=np.uint64)
            for file in files:
                if np.intersect1d(got_tids, file.train_ids).size > 0:
                    sources_with_conflicts.add(source)
                    break
                got_tids = np.union1d(got_tids, file.train_ids)

        if sources_with_conflicts:
            raise ValueError("{} sources have conflicting data "
                             "(same train ID in different files): {}".format(
                len(sources_with_conflicts), ", ".join(sources_with_conflicts)
            ))

    def _expand_trainids(self, counts, trainIds):
        n = min(len(counts), len(trainIds))
        return np.repeat(trainIds[:n], counts.astype(np.intp)[:n])

    def _find_data_chunks(self, source, key):
        """Find contiguous chunks of data for the given source & key

        Yields DataChunk objects.
        """
        if source in self.instrument_sources:
            key_group = key.partition('.')[0]
        else:
            key_group = ''

        for file in self._source_index[source]:
            file_has_data = False
            firsts, counts = file.get_index(source, key_group)

            # Of trains in this file, which are in selection
            selected = np.isin(file.train_ids, self.train_ids)

            # Assemble contiguous chunks of data from this file
            for _from, _to in contiguous_regions(selected):
                file_has_data = True
                yield DataChunk(file, source, key,
                                first=firsts[_from],
                                train_ids=file.train_ids[_from:_to],
                                counts=counts[_from:_to],
                                )

            if not file_has_data:
                # Make an empty chunk to let e.g. get_array find the shape
                yield DataChunk(file, source, key,
                                first=np.uint64(0),
                                train_ids=file.train_ids[:0],
                                counts=counts[:0],
                                )

    def _find_data(self, source, train_id) -> (FileAccess, int):
        for f in self._source_index[source]:
            ixs = (f.train_ids == train_id).nonzero()[0]
            if ixs.size > 0:
                return f, ixs[0]

        return None, None

    def info(self, details_for_sources=()):
        """Show information about the selected data.
        """
        details_sources_re = [re.compile(fnmatch.translate(p))
                              for p in details_for_sources]

        # time info
        train_count = len(self.train_ids)
        if train_count == 0:
            first_train = last_train = '-'
            span_txt = '0.0'
        else:
            first_train = self.train_ids[0]
            last_train = self.train_ids[-1]
            seconds, deciseconds = divmod((last_train - first_train + 1), 10)
            span_txt = '{}.{}'.format(datetime.timedelta(seconds=seconds),
                                      int(deciseconds))

        detector_modules = {}
        for source in self.detector_sources:
            name, modno = DETECTOR_SOURCE_RE.match(source).groups((1, 2))
            detector_modules[(name, modno)] = source

        # A run should only have one detector, but if that changes, don't hide it
        detector_name = ','.join(sorted(set(k[0] for k in detector_modules)))

        # disp
        print('# of trains:   ', train_count)
        print('Duration:      ', span_txt)
        print('First train ID:', first_train)
        print('Last train ID: ', last_train)
        print()

        print("{} detector modules ({})".format(
            len(self.detector_sources), detector_name
        ))
        if len(detector_modules) > 0:
            # Show detail on the first module (the others should be similar)
            mod_key = sorted(detector_modules)[0]
            mod_source = detector_modules[mod_key]
            dinfo = self.detector_info(mod_source)
            module = ' '.join(mod_key)
            dims = ' x '.join(str(d) for d in dinfo['dims'])
            print("  e.g. module {} : {} pixels".format(module, dims))
            print("  {}".format(mod_source))
            print("  {} frames per train, up to {} frames total".format(
                dinfo['frames_per_train'], dinfo['total_frames']
            ))
        print()

        def src_data_detail(s, keys, prefix=''):
            """Detail for how much data is present for an instrument group"""
            if not keys:
                return
            counts = self.get_data_counts(s, list(keys)[0])
            ntrains_data = (counts > 0).sum()
            print(
                f'{prefix}data for {ntrains_data} trains '
                f'({ntrains_data / train_count:.2%}), '
                f'up to {counts.max()} entries per train'
            )

        def keys_detail(s, keys, prefix=''):
            """Detail for a group of keys"""
            for k in keys:
                entry_shape = self.get_entry_shape(s, k)
                if entry_shape:
                    entry_info = f", entry shape {entry_shape}"
                else:
                    entry_info = ""
                dt = self.get_dtype(s, k)
                print(f"{prefix}{k}\t[{dt.name}{entry_info}]")

        non_detector_inst_srcs = self.instrument_sources - self.detector_sources
        print(len(non_detector_inst_srcs), 'instrument sources (excluding detectors):')
        for s in sorted(non_detector_inst_srcs):
            print('  -', s)
            if not any(p.match(s) for p in details_sources_re):
                continue

            # Detail for instrument sources:
            for group, keys in groupby(sorted(self.keys_for_source(s)),
                                       key=lambda k: k.split('.')[0]):
                print(f'    - {group}:')
                keys = list(keys)
                src_data_detail(s, keys, prefix='      ')
                keys_detail(s, keys, prefix='      - ')


        print()
        print(len(self.control_sources), 'control sources: (1 entry per train)')
        for s in sorted(self.control_sources):
            print('  -', s)
            if any(p.match(s) for p in details_sources_re):
                # Detail for control sources: list keys
                keys_detail(s, sorted(self.keys_for_source(s)), prefix='    - ')

        print()

    def detector_info(self, source):
        """Get statistics about the detector data.

        Returns a dictionary with keys:
        - 'dims' (pixel dimensions)
        - 'frames_per_train' (estimated from one file)
        - 'total_frames' (estimated assuming all trains have data)
        """
        source_files = self._source_index[source]
        file0 = sorted(source_files, key=lambda fa: fa.filename)[0]

        _, counts = file0.get_index(source, 'image')
        counts = set(np.unique(counts))
        counts.discard(0)

        if len(counts) > 1:
            warn("Varying number of frames per train: %s" % counts)

        if counts:
            fpt = int(counts.pop())
        else:
            fpt = 0

        dims = file0.file['/INSTRUMENT/{}/image/data'.format(source)].shape[-2:]

        return {
            'dims': dims,
            # Some trains have 0 frames; max is the interesting value
            'frames_per_train': fpt,
            'total_frames': fpt * len(self.train_ids),
        }

    def train_info(self, train_id):
        """Show information about a specific train in the run.

        Parameters
        ----------
        train_id: int
            The specific train ID you get details information.

        Raises
        ------
        ValueError
            if `train_id` is not found in the run.
        """
        if train_id not in self.train_ids:
            raise ValueError("train {} not found in run.".format(train_id))
        files = [f for f in self.files if train_id in f.train_ids]
        ctrl = set().union(*[f.control_sources for f in files])
        inst = set().union(*[f.instrument_sources for f in files])

        # disp
        print('Train [{}] information'.format(train_id))
        print('Devices')
        print('\tInstruments')
        [print('\t-', d) for d in sorted(inst)] or print('\t-')
        print('\tControls')
        [print('\t-', d) for d in sorted(ctrl)] or print('\t-')

    def write(self, filename):
        """Write the selected data to a new HDF5 file

        You can choose a subset of the data using methods
        like :meth:`select` and :meth:`select_trains`,
        then use this write it to a new, smaller file.

        The target filename will be overwritten if it already exists.
        """
        from .writer import FileWriter
        FileWriter(filename, self).write()

    def write_virtual(self, filename):
        """Write an HDF5 file with virtual datasets for the selected data.

        This doesn't copy the data, but each virtual dataset provides a view of
        data spanning multiple sequence files, which can be accessed as if it
        had been copied into one big file.

        This is *not* the same as `building virtual datasets to combine
        multi-module detector data
        <https://in.xfel.eu/readthedocs/docs/data-analysis-user-documentation/en/latest/datafiles.html#combining-detector-data-from-multiple-modules>`__.
        See :doc:`agipd_lpd_data` for that.

        Creating and reading virtual datasets requires HDF5 version 1.10.

        The target filename will be overwritten if it already exists.
        """
        from .writer import VirtualFileWriter
        VirtualFileWriter(filename, self).write()

    def get_virtual_dataset(self, source, key, filename=None):
        """Create an HDF5 virtual dataset for a given source & key

        A dataset looks like a multidimensional array, but the data is loaded
        on-demand when you access it. So it's suitable as an
        interface to data which is too big to load entirely into memory.

        This returns an h5py.Dataset object. This exists in a real file as a
        'virtual dataset', a collection of links pointing to the data in real
        datasets. If *filename* is passed, the file is written at that path,
        overwriting if it already exists. Otherwise, it uses a new temp file.

        To access the dataset from other worker processes, give them the name
        of the created file along with the path to the dataset inside it
        (accessible as ``ds.name``). They will need at least HDF5 1.10 to access
        the virtual dataset, and they must be on a system with access to the
        original data files, as the virtual dataset points to those.
        """
        self._check_field(source, key)

        from .writer import VirtualFileWriter

        if filename is None:
            # Make a temp file to hold the virtual dataset.
            fd, filename = tempfile.mkstemp(suffix='-karabo-data-vds.h5')
            os.close(fd)

        vfw = VirtualFileWriter(filename, self)

        vfw.write_train_ids()

        ds_path = vfw.add_dataset(source, key)

        vfw.write_indexes()
        vfw.write_metadata()
        vfw.set_writer()
        vfw.file.close()  # Close the file for writing and reopen read-only

        f = h5py.File(filename, 'r')
        return f[ds_path]


class TrainIterator:
    """Iterate over trains in a collection of data

    Created by :meth:`DataCollection.trains`.
    """
    def __init__(self, data, require_all=True):
        self.data = data
        self.require_all = require_all
        # {(source, key): (f, dataset)}
        self._datasets_cache = {}

    def _find_data(self, source, key, tid):
        file, ds = self._datasets_cache.get((source, key), (None, None))
        if ds:
            ixs = (file.train_ids == tid).nonzero()[0]
            if ixs.size > 0:
                return file, ixs[0], ds

        data = self.data
        section = 'CONTROL' if source in data.control_sources else 'INSTRUMENT'
        path = '/{}/{}/{}'.format(section, source, key.replace('.', '/'))
        f, pos = data._find_data(source, tid)
        if f is not None:
            ds = f.file[path]
            self._datasets_cache[(source, key)] = (f, ds)
            return f, pos, ds

        return None, None, None

    def _assemble_data(self, tid):
        res = {}
        for source in self.data.control_sources:
            source_data = res[source] = {
                'metadata': {'source': source, 'timestamp.tid': tid}
            }
            for key in self.data.keys_for_source(source):
                _, pos, ds = self._find_data(source, key, tid)
                if ds is None:
                    continue
                source_data[key] = ds[pos]

        for source in self.data.instrument_sources:
            source_data = res[source] = {
                'metadata': {'source': source, 'timestamp.tid': tid}
            }
            for key in self.data.keys_for_source(source):
                file, pos, ds = self._find_data(source, key, tid)
                if ds is None:
                    continue
                group = key.partition('.')[0]
                firsts, counts = file.get_index(source, group)
                first, count = firsts[pos], counts[pos]
                if count == 1:
                    source_data[key] = ds[first]
                else:
                    source_data[key] = ds[first : first + count]

        return res

    def __iter__(self):
        for tid in self.data.train_ids:
            tid = int(tid)  # Convert numpy int to regular Python int
            if self.require_all and self.data._check_data_missing(tid):
                continue
            yield tid, self._assemble_data(tid)


def H5File(path):
    """Open a single HDF5 file generated at European XFEL.

    ::

        file = H5File("RAW-R0017-DA01-S00000.h5")

    Returns a :class:`DataCollection` object.

    Parameters
    ----------
    path: str
        Path to the HDF5 file
    """
    return DataCollection.from_path(path)


def RunDirectory(path, include='*', file_filter=locality.lc_any):
    """Open data files from a 'run' at European XFEL.

    ::

        run = RunDirectory("/gpfs/exfel/exp/XMPL/201750/p700000/raw/r0001")

    A 'run' is a directory containing a number of HDF5 files with data from the
    same time period.

    Returns a :class:`DataCollection` object.

    Parameters
    ----------
    path: str
        Path to the run directory containing HDF5 files.
    include: str
        Wildcard string to filter data files.
    file_filter: callable
        Function to subset the list of filenames to open.
        Meant to be used with functions in the extra_data.locality module.
    """
    files = [f for f in os.listdir(path) if f.endswith('.h5')]
    files = [osp.join(path, f) for f in fnmatch.filter(files, include)]
    files = file_filter(files)
    if not files:
        raise Exception("No HDF5 files found in {} with glob pattern {}".format(path, include))

    files_map = RunFilesMap(path)
    t0 = time.monotonic()
    d = DataCollection.from_paths(files, files_map)
    log.debug("Opened run with %d files in %.2g s",
              len(d.files), time.monotonic() - t0)
    files_map.save(d.files)

    return d

# RunDirectory was previously RunHandler; we'll leave it accessible in case
# any code was already using it.
RunHandler = RunDirectory


def open_run(proposal, run, data='raw', include='*', file_filter=locality.lc_any):
    """Access EuXFEL data on the Maxwell cluster by proposal and run number.

    ::

        run = open_run(proposal=700000, run=1)

    Returns a :class:`DataCollection` object.

    Parameters
    ----------
    proposal: str, int
        A proposal number, such as 2012, '2012', 'p002012', or a path such as
        '/gpfs/exfel/exp/SPB/201701/p002012'.
    run: str, int
        A run number such as 243, '243' or 'r0243'.
    data: str
        'raw' or 'proc' (processed) to access data from one of those folders.
        The default is 'raw'.
    include: str
        Wildcard string to filter data files.
    file_filter: callable
        Function to subset the list of filenames to open.
        Meant to be used with functions in the extra_data.locality module.
    """
    if isinstance(proposal, str):
        if ('/' not in proposal) and not proposal.startswith('p'):
            proposal = 'p' + proposal.rjust(6, '0')
    else:
        # Allow integers, including numpy integers
        proposal = 'p{:06d}'.format(index(proposal))

    prop_dir = find_proposal(proposal)

    if isinstance(run, str):
        if run.startswith('r'):
            run = run[1:]
    else:
        run = index(run)  # Allow integers, including numpy integers
    run = 'r' + str(run).zfill(4)

    return RunDirectory(
        osp.join(prop_dir, data, run), include=include, file_filter=file_filter
    )
