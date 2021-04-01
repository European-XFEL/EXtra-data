from typing import List, Optional

import numpy as np

from .exceptions import TrainIDError
from .file_access import FileAccess
from .read_machinery import contiguous_regions, DataChunk, select_train_ids

class KeyData:
    """Data for one key in one source

    Don't create this directly; get it from ``run[source, key]``.
    """
    def __init__(
            self, source, key, *, train_ids, files, section, dtype, eshape,
            inc_suspect_trains=False,
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
        for file in self.files:
            firsts, counts = file.get_index(self.source, self._key_group)

            # Of trains in this file, which are in selection
            include = np.isin(file.train_ids, self.train_ids)
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
                [c for c in self._find_chunks() if c.total_count],
                key=lambda c: c.train_ids[0]
            )
        return self._cached_chunks

    def __repr__(self):
        return f"<extra_data.KeyData source={self.source!r} key={self.key!r} " \
               f"for {len(self.train_ids)} trains"

    @property
    def _key_group(self):
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

    def select_trains(self, trains):
        """Select a subset of trains in this data as a new :class:`KeyData` object.

        Also available by slicing and indexing the KeyData object::

            run[source, key][:10]  # Select data for first 10 trains
        """
        tids = select_train_ids(self.train_ids, trains)
        files = [f for f in self.files
                 if np.intersect1d(f.train_ids, tids).size > 0]

        return KeyData(
            self.source,
            self.key,
            train_ids=tids,
            files=files,
            section=self.section,
            dtype=self.dtype,
            eshape=self.entry_shape,
        )

    def __getitem__(self, item):
        return self.select_trains(item)

    def data_counts(self):
        """Get a count of data entries in each train.

        Returns a pandas series with an index of train IDs.
        """
        import pandas as pd

        seq_series = []

        for f in self.files:
            if self.section == 'CONTROL':
                counts = np.ones(len(f.train_ids), dtype=np.uint64)
            else:
                _, counts = f.get_index(self.source, self._key_group)

            if self.inc_suspect_trains:
                s = pd.Series(counts, index=f.train_ids)
            else:
                s = pd.Series(counts[f.validity_flag], index=f.valid_train_ids)
            seq_series.append(s)

        ser = pd.concat(sorted(seq_series, key=lambda s: s.index[0]))
        # Select out only the train IDs of interest
        train_ids = ser.index.intersection(np.asarray(self.train_ids))
        return ser.loc[train_ids]

    # Getting data as different kinds of array: -------------------------------

    def ndarray(self, roi=()):
        """Load this data as a numpy array

        *roi* may be a ``numpy.s_[]`` expression to load e.g. only part of each
        image from a camera.
        """
        if not isinstance(roi, tuple):
            roi = (roi,)

        # Find the shape of the array with the ROI applied
        roi_dummy = np.zeros((0,) + self.entry_shape) # extra 0 dim: use less memory
        roi_shape = roi_dummy[np.index_exp[:] + roi].shape[1:]

        out = np.empty(self.shape[:1] + roi_shape, dtype=self.dtype)

        # Read the data from each chunk into the result array
        dest_cursor = 0
        for chunk in self._data_chunks:
            dest_chunk_end = dest_cursor + chunk.total_count

            slices = (chunk.slice,) + roi
            chunk.dataset.read_direct(
                out[dest_cursor:dest_chunk_end], source_sel=slices
            )
            dest_cursor = dest_chunk_end

        return out

    def _trainid_index(self):
        """A 1D array of train IDs, corresponding to self.shape[0]"""
        chunks_trainids = [
            np.repeat(chunk.train_ids, chunk.counts.astype(np.intp))
            for chunk in self._data_chunks
        ]
        return np.concatenate(chunks_trainids)

    def xarray(self, extra_dims=None, roi=(), name=None):
        """Load this data as a labelled xarray.DataArray.

        The first dimension is labelled with train IDs. Other dimensions may be
        named by passing a list of names to *extra_dims*.

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
            by a dot.
        """
        import xarray

        ndarr = self.ndarray(roi=roi)

        # Dimension labels
        if extra_dims is None:
            extra_dims = ['dim_%d' % i for i in range(ndarr.ndim - 1)]
        dims = ['trainId'] + extra_dims

        # Train ID index
        coords = {}
        if self.shape[0]:
            coords = {'trainId': self._trainid_index()}

        if name is None:
            name = f'{self.source}.{self.key}'

            if name.endswith('.value') and self.section == 'CONTROL':
                name = name[:-6]

        return xarray.DataArray(ndarr, dims=dims, coords=coords, name=name)

    def series(self):
        """Load this data as a pandas Series. Only for 1D data.
        """
        import pandas as pd

        if self.ndim > 1:
            raise TypeError("pandas Series are only available for 1D data")

        name = self.source + '/' + self.key
        if name.endswith('.value') and self.section == 'CONTROL':
            name = name[:-6]

        index = pd.Index(self._trainid_index(), name='trainId')
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

        for chunk in self._data_chunks:
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
            while np.product(chunk_shape) * itemsize > limit and chunk_dim0 > 1:
                chunk_dim0 //= 2
                chunk_shape = (chunk_dim0,) + chunk.dataset.shape[1:]

            chunks_darrs.append(
                da.from_array(
                    chunk.file.dset_proxy(chunk.dataset_path), chunks=chunk_shape
                )[chunk.slice]
            )

        dask_arr = da.concatenate(chunks_darrs, axis=0)

        if labelled:
            # Dimension labels
            dims = ['trainId'] + ['dim_%d' % i for i in range(dask_arr.ndim - 1)]

            # Train ID index
            coords = {'trainId': self._trainid_index()}

            import xarray
            return xarray.DataArray(dask_arr, dims=dims, coords=coords)
        else:
            return dask_arr

    # Getting data by train: --------------------------------------------------

    def _find_tid(self, tid) -> (Optional[FileAccess], int):
        for fa in self.files:
            matches = (fa.train_ids == tid).nonzero()[0]
            if matches.size > 0:
                return fa, matches[0]

        return None, 0

    def train_from_id(self, tid):
        """Get data for the given train ID as a numpy array.

        Returns (train ID, array)
        """
        if tid not in self.train_ids:
            raise TrainIDError(tid)

        fa, ix = self._find_tid(tid)
        if fa is None:
            return np.empty((0,) + self.entry_shape, dtype=self.dtype)

        firsts, counts = fa.get_index(self.source, self._key_group)
        first, count = firsts[ix], counts[ix]
        if count == 1:
            return tid, fa.file[self.hdf5_data_path][first]
        else:
            return tid, fa.file[self.hdf5_data_path][first: first+count]

    def train_from_index(self, i):
        """Get data for a train by index (starting at 0) as a numpy array.

        Returns (train ID, array)
        """
        return self.train_from_id(self.train_ids[i])

    def trains(self):
        """Iterate through trains containing data for this key

        Yields pairs of (train ID, array). Skips trains where data is missing.
        """
        for chunk in self._data_chunks:
            start = chunk.first
            ds = chunk.dataset
            for tid, count in zip(chunk.train_ids, chunk.counts):
                if count > 1:
                    yield tid, ds[start: start+count]
                elif count == 1:
                    yield tid, ds[start]

                start += count
