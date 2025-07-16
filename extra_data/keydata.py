from typing import List, Optional, Tuple

import h5py
import numpy as np

from .exceptions import TrainIDError, NoDataError
from .file_access import FileAccess
from .read_machinery import (
    contiguous_regions, DataChunk, select_train_ids, split_trains, roi_shape,
    trains_files_index,
)

class KeyData:
    """Data for one key in one source

    Don't create this directly; get it from ``run[source, key]``.
    """
    def __init__(
            self, source, key, *, train_ids, files, section, dtype, eshape,
            inc_suspect_trains=True,
    ):
        self.source = source
        self.key = key
        self.train_ids = train_ids
        self.files: List[FileAccess] = files
        self.section = section
        self.dtype = dtype
        self.entry_shape = eshape
        self.ndim = len(eshape) + 1
        self.inc_suspect_trains = inc_suspect_trains

    def _find_chunks(self):
        """Find contiguous chunks of data for this key, in any order."""
        all_tids_arr = np.array(self.train_ids)

        for file in self.files:
            if len(file.train_ids) == 0:
                continue

            firsts, counts = file.get_index(self.source, self.index_group)

            # Of trains in this file, which are in selection
            include = np.isin(file.train_ids, all_tids_arr)
            if not self.inc_suspect_trains:
                include &= file.validity_flag

            # Assemble contiguous chunks of data from this file
            for _from, _to in contiguous_regions(include):
                yield DataChunk(
                    file, self.hdf5_data_path,
                    first=firsts[_from],
                    train_ids=file.train_ids[_from:_to],
                    counts=counts[_from:_to],
                )

    _cached_chunks = None

    @property
    def _data_chunks(self) -> List[DataChunk]:
        """An ordered list of chunks containing data"""
        if self._cached_chunks is None:
            self._cached_chunks = sorted(
                self._find_chunks(), key=lambda c: c.train_ids[0]
            )
        return self._cached_chunks

    @property
    def _data_chunks_nonempty(self) -> List[DataChunk]:
        return [c for c in self._data_chunks if c.total_count]

    def __repr__(self):
        return f"<extra_data.KeyData source={self.source!r} key={self.key!r} " \
               f"for {len(self.train_ids)} trains>"

    @property
    def is_control(self):
        """Whether this key belongs to a control source."""
        return self.section == 'CONTROL'

    @property
    def is_instrument(self):
        """Whether this key belongs to an instrument source."""
        return self.section == 'INSTRUMENT'

    @property
    def index_group(self):
        """The part of the key needed to look up index data"""
        if self.section == 'INSTRUMENT':
            return self.key.partition('.')[0]
        else:
            return ''

    @property
    def hdf5_data_path(self):
        """The path to the relevant dataset within each HDF5 file"""
        return f"/{self.section}/{self.source}/{self.key.replace('.', '/')}"

    @property
    def shape(self):
        """The shape of this data as a tuple, like for a NumPy array.

        Finding the shape may require getting index data from several files
        """
        return (sum(c.total_count for c in self._data_chunks),) + self.entry_shape

    @property
    def nbytes(self):
        """The number of bytes this data would take up in memory."""
        return self.dtype.itemsize * np.prod(self.shape)

    @property
    def size_mb(self):
        """The size of the data in memory in megabytes."""
        return self.nbytes / 1e6

    @property
    def size_gb(self):
        """The size of the data in memory in gigabytes."""
        return self.nbytes / 1e9

    @property
    def units(self):
        """The units symbol for this data, e.g. 'μJ', or None if not found"""
        attrs = self.attributes()
        base_unit = attrs.get('unitSymbol', None)
        if base_unit is None:
            return None

        prefix = attrs.get('metricPrefixSymbol', '')
        if prefix == 'u':
            prefix = 'μ'  # We are not afraid of unicode
        return prefix + base_unit

    @property
    def units_name(self):
        """The units name for this data, e.g. 'microjoule', or None if not found"""
        attrs = self.attributes()
        base_unit = attrs.get('unitName', None)
        if base_unit is None:
            return None

        prefix = attrs.get('metricPrefixName', '')
        return prefix + base_unit

    @property
    def source_file_paths(self):
        paths = dict()
        for chunk in self._data_chunks:
            paths |= dict.fromkeys(chunk.source_file_paths)
        paths = list(paths.keys())

        # Fallback for virtual overview files where no data was recorded for
        # this source, so there's no mapping to point to.
        if not paths:
            source_grp = self.files[0].file[f"{self.section}/{self.source}"]
            if 'source_files' in source_grp.attrs:
                paths.append(source_grp.attrs['source_files'][0])

        from pathlib import Path
        return [Path(p) for p in paths]

    def _without_virtual_overview(self):
        if not self.files[0].file[self.hdf5_data_path].is_virtual:
            # We're already looking at regular source files
            return self

        return KeyData(
            self.source, self.key,
            train_ids=self.train_ids,
            files=[FileAccess(p) for p in self.source_file_paths],
            section=self.section,
            dtype=self.dtype,
            eshape=self.entry_shape,
            inc_suspect_trains=self.inc_suspect_trains,
        )

    def _find_attributes(self, dset):
        """Find Karabo attributes belonging to a dataset."""
        attrs = dict(dset.attrs)

        if self.is_control and self.key.endswith('.value'):
            # For CONTROL sources, most of the attributes are saved on
            # the parent group rather than the .value dataset. In the
            # case of duplicated keys, the parent value appears to be
            # the correct one.
            attrs.update(dict(dset.parent.attrs))

        return attrs

    def attributes(self):
        """Get a dict of all attributes stored with this data

        This may be awkward to use. See .units and .units_name for more
        convenient forms.
        """
        dset = self.files[0].file[self.hdf5_data_path]
        attrs = self._find_attributes(dset)
        if (not attrs) and dset.is_virtual:
            # Virtual datasets were initially created without these attributes.
            # Find a source file. Not using source_file_paths as it can give [].
            _, filename, _, _ = dset.virtual_sources()[0]
            # Not using FileAccess: no need for train or source lists.
            with h5py.File(filename, 'r') as f:
                attrs = self._find_attributes(f[self.hdf5_data_path])

        return attrs

    def select_trains(self, trains):
        """Select a subset of trains in this data as a new :class:`KeyData` object.

        Also available by slicing and indexing the KeyData object::

            run[source, key][:10]  # Select data for first 10 trains
        """
        tids = select_train_ids(self.train_ids, trains)
        return self._only_tids(tids)

    def __getitem__(self, item):
        return self.select_trains(item)

    __iter__ = None  # Disable iteration

    def _only_tids(self, tids, files=None):
        tids_arr = np.array(tids)
        if files is None:
            files = [
                f for f in self.files
                if f.has_train_ids(tids_arr, self.inc_suspect_trains)
            ]
        if not files:
            # Keep 1 file, even if 0 trains selected.
            files = [self.files[0]]

        return KeyData(
            self.source,
            self.key,
            train_ids=tids,
            files=files,
            section=self.section,
            dtype=self.dtype,
            eshape=self.entry_shape,
            inc_suspect_trains=self.inc_suspect_trains,
        )

    def drop_empty_trains(self):
        """Select only trains with data as a new :class:`KeyData` object."""
        counts = self.data_counts(labelled=False)
        tids = np.array(self.train_ids)[counts > 0]
        return self._only_tids(list(tids))

    def split_trains(self, parts=None, trains_per_part=None):
        """Split this data into chunks with a fraction of the trains each.

        Either *parts* or *trains_per_part* must be specified.

        This returns an iterator yielding new :class:`KeyData` objects.
        The parts will have similar sizes, e.g. splitting 11 trains
        with ``trains_per_part=8`` will produce 5 & 6 trains, not 8 & 3.
        Selected trains count even if they are missing data, so different
        keys from the same run can be split into matching chunks.

        Parameters
        ----------

        parts: int
            How many parts to split the data into. If trains_per_part is also
            specified, this is a minimum, and it may make more parts.
            It may also make fewer if there are fewer trains in the data.
        trains_per_part: int
            A maximum number of trains in each part. Parts will often have
            fewer trains than this.
        """
        # tids_files points to the file for each train.
        # This avoids checking all files for each chunk, which can be slow.
        tids_files = trains_files_index(
            self.train_ids, self.files, self.inc_suspect_trains
        )
        for sl in split_trains(len(self.train_ids), parts, trains_per_part):
            tids = self.train_ids[sl]
            files = set(tids_files[sl]) - {None}
            files = sorted(files, key=lambda f: f.filename)
            yield self._only_tids(tids, files=files)

    def data_counts(self, labelled=True):
        """Get a count of data entries in each train.

        If *labelled* is True, returns a pandas series with an index of train
        IDs. Otherwise, returns a NumPy array of counts to match ``.train_ids``.
        """
        if self._data_chunks:
            train_ids = np.concatenate([c.train_ids for c in self._data_chunks])
            counts = np.concatenate([c.counts for c in self._data_chunks])
        else:
            train_ids = counts = np.zeros(0, dtype=np.uint64)

        if labelled:
            import pandas as pd
            return pd.Series(counts, index=train_ids)
        else:
            all_tids_arr = np.array(self.train_ids)
            res = np.zeros(len(all_tids_arr), dtype=np.uint64)
            tid_to_ix = np.intersect1d(all_tids_arr, train_ids, return_indices=True)[1]

            # We may be missing some train IDs, if they're not in any file
            # for this source, and they're sometimes out of order within chunks
            # (they shouldn't be, but we try not to fail too badly if they are).
            assert len(tid_to_ix) == len(train_ids)
            res[tid_to_ix] = counts

            return res

    def as_single_value(self, rtol=1e-5, atol=0.0, reduce_by=None):
        """Retrieve a single reduced value if within tolerances.

        The relative and absolute tolerances *rtol* and *atol* work the
        same way as in ``numpy.allclose``. The default relative tolerance
        is 1e-5 with no absolute tolerance. The data for this key is compared
        against a reduced value obtained by the method described in *reduce_by*.

        This may be a callable taking the key data, the string value of a
        global symbol in the numpy packge such as 'median' or 'first' to use
        the first value encountered. By default, 'median' is used.

        If within tolerances, the reduced value is returned.

        For non-numerical keys like strings, the method instead always
        checks for uniqueness and returns such a value, if present.
        """

        data = self.ndarray()

        if len(data) == 0:
            raise NoDataError(self.source, self.key)

        if not np.issubdtype(self.dtype, np.number):
            # Handle non-numeric types first.

            if reduce_by is not None:
                raise TypeError('custom reduce method not supported for '
                                'non-numeric type')

            unique_values = np.unique(data, axis=None)

            if len(unique_values) > 1:
                raise ValueError(f'str values are not unique: {unique_values}')

            return unique_values[0]

        elif reduce_by is None:
            reduce_by = 'median'

        if callable(reduce_by):
            value = reduce_by(data)
        elif isinstance(reduce_by, str) and hasattr(np, reduce_by):
            value = getattr(np, reduce_by)(data, axis=0)
        elif reduce_by == 'first':
            value = data[0]
        else:
            raise ValueError('invalid reduction method (may be callable, '
                             'global numpy symbol or "first")')

        if not np.allclose(data, value, rtol=rtol, atol=atol):
            adev = np.max(np.abs(data - value))
            rdev = np.max(np.abs(adev / value))

            raise ValueError(f'data values are not within tolerance '
                             f'(absolute: {adev:.3g}, relative: {rdev:.3g})')

        return value

    # Getting data as different kinds of array: -------------------------------

    def ndarray(self, roi=(), out=None):
        """Load this data as a numpy array

        *roi* may be a ``numpy.s_[]`` expression to load e.g. only part of each
        image from a camera. If *out* is not given, a suitable array will be
        allocated.
        """
        if not isinstance(roi, tuple):
            roi = (roi,)

        req_shape = self.shape[:1] + roi_shape(self.entry_shape, roi)

        if out is None:
            out = np.empty(req_shape, dtype=self.dtype)
        elif out is not None and out.shape != req_shape:
            raise ValueError(f'requires output array of shape {req_shape}')

        # Read the data from each chunk into the result array
        dest_cursor = 0
        for chunk in self._data_chunks_nonempty:
            dest_chunk_end = dest_cursor + chunk.total_count

            slices = (chunk.slice,) + roi
            chunk.dataset.read_direct(
                out[dest_cursor:dest_chunk_end], source_sel=slices
            )
            dest_cursor = dest_chunk_end

        if out.dtype.hasobject:
            # Can current only occur for string properties, convert from
            # object array of bytes to to object array of strings.
            # This will fail for structured dtypes containing strings,
            # but such are not known to us yet.
            out = np.array(
                [x.decode('utf8', 'surrogateescape') for x in out.flat],
                dtype=object
            ).reshape(out.shape)

        return out

    def train_id_coordinates(self):
        """Make an array of train IDs to use alongside data from ``.ndarray()``.

        :attr:`train_ids` includes each selected train ID once, including trains
        where data is missing. :meth:`train_id_coordinates` excludes missing
        trains, and repeats train IDs if the source has multiple entries
        per train. The result will be the same length as the first dimension
        of an array from :meth:`ndarray`, and tells you which train each entry
        belongs to.

        .. seealso::

           :meth:`xarray` returns a labelled array including these train IDs.
        """
        if not self._data_chunks:
            return np.zeros(0, dtype=np.uint64)
        chunks_trainids = [
            np.repeat(chunk.train_ids, chunk.counts.astype(np.intp))
            for chunk in self._data_chunks
        ]
        return np.concatenate(chunks_trainids)

    def xarray(self, extra_dims=None, roi=(), name=None):
        """Load this data as a labelled xarray array or dataset.

        The first dimension is labelled with train IDs. Other dimensions
        may be named by passing a list of names to *extra_dims*.

        For scalar datatypes, an xarray.DataArray is returned using either
        the supplied *name* or the concatenated source and key name if omitted.

        If the data is stored in a structured datatype, an xarray.Dataset
        is returned with a variable for each field. The data of these
        variables will be non-contiguous in memory, use
        `Dataset.copy(deep=true)` to obtain a contiguous copy.

        Parameters
        ----------

        extra_dims: list of str
            Name extra dimensions in the array. The first dimension is
            automatically called 'train'. The default for extra dimensions
            is dim_0, dim_1, ...
        roi: numpy.s_[], slice, tuple of slices, or by_index
            The region of interest. This expression selects data in all
            dimensions apart from the first (trains) dimension. If the data
            holds a 1D array for each entry, roi=np.s_[:8] would get the
            first 8 values from every train. If the data is 2D or more at
            each entry, selection looks like roi=np.s_[:8, 5:10] .
        name: str
            Name the array itself. The default is the source and key joined
            by a dot. Ignored for structured data when a dataset is returned.
        """
        import xarray

        ndarr = self.ndarray(roi=roi)

        # Dimension labels
        if extra_dims is None:
            extra_dims = ['dim_%d' % i for i in range(ndarr.ndim - 1)]
        dims = ['trainId'] + extra_dims

        # Train ID index
        coords = {'trainId': self.train_id_coordinates()}
        # xarray attributes
        attrs = {}
        if (units := self.units):
            attrs['units'] = units

        if ndarr.dtype.names is not None:
            # Structured dtype.
            return xarray.Dataset(
                {field: (dims, ndarr[field]) for field in ndarr.dtype.names},
                coords=coords, attrs=attrs)
        else:
            if name is None:
                name = f'{self.source}.{self.key}'

                if name.endswith('.value') and self.section == 'CONTROL':
                    name = name[:-6]

            # Primitive dtype.
            return xarray.DataArray(
                ndarr, dims=dims, coords=coords, name=name, attrs=attrs)

    def series(self):
        """Load this data as a pandas Series. Only for 1D data.
        """
        import pandas as pd

        if self.ndim > 1:
            raise TypeError("pandas Series are only available for 1D data")

        name = self.source + '/' + self.key
        if name.endswith('.value') and self.section == 'CONTROL':
            name = name[:-6]

        index = pd.Index(self.train_id_coordinates(), name='trainId')
        data = self.ndarray()
        return pd.Series(data, name=name, index=index)

    def dask_array(self, labelled=False):
        """Make a Dask array for this data.

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

        labelled: bool
            If True, label the train IDs for the data, returning an
            xarray.DataArray object wrapping a Dask array.
        """
        import dask.array as da

        chunks_darrs = []

        for chunk in self._data_chunks_nonempty:
            chunk_dim0 = chunk.total_count
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
            while np.prod(chunk_shape) * itemsize > limit and chunk_dim0 > 1:
                chunk_dim0 //= 2
                chunk_shape = (chunk_dim0,) + chunk.dataset.shape[1:]

            chunks_darrs.append(
                da.from_array(
                    chunk.file.dset_proxy(chunk.dataset_path), chunks=chunk_shape
                )[chunk.slice]
            )

        if chunks_darrs:
            dask_arr = da.concatenate(chunks_darrs, axis=0)
        else:
            shape = (0,) + self.entry_shape
            dask_arr = da.zeros(shape=shape, dtype=self.dtype, chunks=shape)

        if labelled:
            # Dimension labels
            dims = ['trainId'] + ['dim_%d' % i for i in range(dask_arr.ndim - 1)]

            # Train ID index
            coords = {'trainId': self.train_id_coordinates()}

            import xarray
            return xarray.DataArray(dask_arr, dims=dims, coords=coords)
        else:
            return dask_arr

    # Getting data by train: --------------------------------------------------

    def _find_tid(self, tid) -> Tuple[Optional[FileAccess], int]:
        for fa in self.files:
            matches = (fa.train_ids == tid).nonzero()[0]
            if self.inc_suspect_trains and matches.size > 0:
                return fa, matches[0]

            for ix in matches:
                if fa.validity_flag[ix]:
                    return fa, ix

        return None, 0

    def train_from_id(self, tid, keep_dims=False):
        """Get data for the given train ID as a numpy array.

        Returns (train ID, array)
        """
        if tid not in self.train_ids:
            raise TrainIDError(tid)

        fa, ix = self._find_tid(tid)
        if fa is None:
            return np.empty((0,) + self.entry_shape, dtype=self.dtype)

        firsts, counts = fa.get_index(self.source, self.index_group)
        first, count = firsts[ix], counts[ix]
        if count == 1 and not keep_dims:
            return tid, fa.file[self.hdf5_data_path][first]
        else:
            return tid, fa.file[self.hdf5_data_path][first: first+count]

    def train_from_index(self, i, keep_dims=False):
        """Get data for a train by index (starting at 0) as a numpy array.

        Returns (train ID, array)
        """
        return self.train_from_id(self.train_ids[i], keep_dims=keep_dims)

    def trains(self, keep_dims=False, include_empty=False):
        """Iterate through trains containing data for this key

        Yields pairs of (train ID, array). Train axis is removed in case
        of single elements unless *keep_dims* is set. Skips trains where
        data is missing unless *include_empty* is set, returning None or
        zero-length array with *keep_dims*.
        """
        if keep_dims and include_empty:
            empty_result = np.zeros(shape=(0,) + self.entry_shape,
                                    dtype=self.dtype)
        else:
            empty_result = None

        for chunk in self._data_chunks_nonempty:
            start = chunk.first
            ds = chunk.dataset
            for tid, count in zip(chunk.train_ids, chunk.counts):
                if count > 1 or keep_dims:
                    yield tid, ds[start: start+count]
                elif count == 1:
                    yield tid, ds[start]
                elif include_empty:
                    yield tid, empty_result

                start += count
