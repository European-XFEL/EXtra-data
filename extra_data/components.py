"""Interfaces to data from specific instruments
"""
import logging
import numpy as np
import pandas as pd
import re
import xarray

from .exceptions import SourceNameError
from .reader import DataCollection, by_id, by_index
from .read_machinery import roi_shape, DataChunk
from .writer import FileWriter
from .write_cxi import XtdfCXIWriter, JUNGFRAUCXIWriter

__all__ = [
    'AGIPD1M',
    'AGIPD500K',
    'DSSC1M',
    'LPD1M',
    'JUNGFRAU',
    'identify_multimod_detectors',
]

log = logging.getLogger(__name__)

MAX_PULSES = 2700
NO_PULSE_ID = 9999


def multimod_detectors(detector_cls):
    """
    Decorator for multimod detector classes (e.g. AGIPD/LPD/JUNGFRAU)
    to store them in a list 'multimod_detectors.list' and their names
    in 'multimod_detectors.names'.

    Parameters
    ----------
    detector_cls: class
      Decorated detector class to append to the list.

    Returns
    -------
    detector_cls: class
      Unmodified decorated detector class.
    """
    multimod_detectors.list = getattr(multimod_detectors, 'list', list())
    multimod_detectors.list.append(detector_cls)
    multimod_detectors.names = getattr(multimod_detectors, 'names', list())
    multimod_detectors.names.append(detector_cls.__name__)
    return detector_cls


def _check_pulse_selection(pulses):
    """Check and normalise a pulse selection"""
    if not isinstance(pulses, (by_id, by_index)):
        pulses = by_index[pulses]

    val = pulses.value

    if isinstance(pulses.value, slice):
        # Ensure start/stop/step are all real numbers
        start = val.start if (val.start is not None) else 0
        stop = val.stop if (val.stop is not None) else MAX_PULSES
        step = val.step if (val.step is not None) else 1

        if not all(isinstance(s, int) for s in (start, stop, step)):
            raise TypeError("Pulse selection slice must use integers or None")
        if step < 1:
            raise ValueError("Pulse selection slice must have positive step")
        if (start < 0) or (stop < 0):
            raise NotImplementedError("Negative pulse indices not supported")

        return type(pulses)(slice(start, stop, step))

    # Convert everything except slices to numpy arrays
    elif isinstance(pulses.value, int):
        val = np.array([val], dtype=np.uint64)
    else:
        val = np.asarray(val, dtype=np.uint64)

    if (val < 0).any():
        if isinstance(pulses, by_id):
            raise ValueError("Pulse IDs cannot be negative")
        else:
            raise NotImplementedError("Negative pulse indices not supported")

    return type(pulses)(val)


class MultimodDetectorBase:
    """Base class for detectors made of several modules as separate data sources
    """

    _source_re = re.compile(r'(?P<detname>.+)/DET/(\d+)CH')
    # Override in subclass
    _main_data_key = ''  # Key to use for checking data counts match
    module_shape = (0, 0)
    n_modules = 0

    def __init__(self, data: DataCollection, detector_name=None, modules=None,
                 *, min_modules=1):
        if detector_name is None:
            detector_name = self._find_detector_name(data)
        if min_modules <= 0:
            raise ValueError("min_modules must be a positive integer, not "
                             f"{min_modules!r}")

        source_to_modno = self._identify_sources(data, detector_name, modules)

        data = data.select([(src, '*') for src in source_to_modno])
        self.detector_name = detector_name
        self.source_to_modno = source_to_modno

        # pandas' missing-data handling converts the data to floats if there
        # are any gaps - so fill them with 0s and convert back to uint64.
        mod_data_counts = pd.DataFrame({
            src: data.get_data_counts(src, self._main_data_key)
            for src in source_to_modno
        }).fillna(0).astype(np.uint64)

        # Within any train, all modules should have same count or zero
        frame_counts = pd.Series(0, index=mod_data_counts.index, dtype=np.uint64)
        for tid, data_counts in mod_data_counts.iterrows():
            count_vals = set(data_counts) - {0}
            if len(count_vals) > 1:
                raise ValueError(
                    f"Inconsistent frame counts for train {tid}: {count_vals}"
                )
            elif count_vals:
                frame_counts[tid] = count_vals.pop()

        self.data = self._select_trains(data, mod_data_counts, min_modules)

        # This should be a reversible 1-to-1 mapping
        self.modno_to_source = {m: s for (s, m) in source_to_modno.items()}
        assert len(self.modno_to_source) == len(self.source_to_modno)

        train_id_arr = np.asarray(self.data.train_ids)
        split_indices = np.where(np.diff(train_id_arr) != 1)[0] + 1
        self.train_id_chunks = np.split(train_id_arr, split_indices)
        self.frame_counts = frame_counts[train_id_arr]

        self.train_ids_perframe = np.repeat(
            self.frame_counts.index.values, self.frame_counts.values.astype(np.intp)
        )

        # Cumulative sum gives the end of each train, subtract to get start
        self.train_id_to_ix = self.frame_counts.cumsum() - self.frame_counts

    @classmethod
    def _find_detector_name(cls, data):
        detector_names = set()
        for source in data.instrument_sources:
            m = cls._source_re.match(source)
            if m:
                detector_names.add(m.group('detname'))
        if not detector_names:
            raise SourceNameError(cls._source_re.pattern)
        elif len(detector_names) > 1:
            raise ValueError(
                "Multiple detectors found in the data: {}. "
                "Pass a name to data.detector() to pick one.".format(
                    ', '.join(repr(n) for n in detector_names)
                )
            )
        return detector_names.pop()

    def _source_matches(self, data, detector_name):
        for source in data.instrument_sources:
            m = self._source_re.match(source)
            if m and m.group('detname') == detector_name:
                yield source, int(m.group('modno'))

    def _identify_sources(self, data, detector_name, modules=None):
        source_to_modno = dict(self._source_matches(data, detector_name))

        if modules is not None:
            source_to_modno = {s: n for (s, n) in source_to_modno.items()
                               if n in modules}

        if not source_to_modno:
            raise SourceNameError(f'{detector_name}/DET/...')

        return source_to_modno

    @classmethod
    def _select_trains(cls, data, mod_data_counts, min_modules):
        modules_present = (mod_data_counts > 0).sum(axis=1)
        mod_data_counts = mod_data_counts[modules_present >= min_modules]
        ntrains = len(mod_data_counts)
        if not ntrains:
            raise ValueError("No data found with >= {} modules present"
                             .format(min_modules))
        log.info("Found %d trains with data for at least %d modules",
                 ntrains, min_modules)
        train_ids = mod_data_counts.index.values
        return data.select_trains(by_id[train_ids])

    @staticmethod
    def _split_align_chunk(chunk, target_train_ids: np.ndarray):
        """
        Split up a source chunk to align with parts of a joined array.

        Chunk points to contiguous source data, but if this misses a train,
        it might not correspond to a contiguous region in the output. This
        yields pairs of (target_slice, source_slice) describing chunks that can
        be copied/mapped to a similar block in the output.

        Parameters
        ----------
        chunk: read_machinery::DataChunk
          Reference to a contiguous chunk of data to be mapped.
        target_train_ids: numpy.ndarray
          Train ID index for target array to align chunk data to. Train IDs may
          occur more than once in here.
        """
        # Expand the list of train IDs to one per frame
        chunk_tids = np.repeat(chunk.train_ids, chunk.counts.astype(np.intp))

        chunk_match_start = int(chunk.first)

        while chunk_tids.size > 0:
            # Look up where the start of this chunk fits in the target
            tgt_start = (target_train_ids == chunk_tids[0]).nonzero()[0][0]

            target_tids = target_train_ids[
                tgt_start : tgt_start + len(chunk_tids)
            ]
            assert target_tids.shape == chunk_tids.shape, \
                f"{target_tids.shape} != {chunk_tids.shape}"
            assert target_tids[0] == chunk_tids[0], \
                f"{target_tids[0]} != {chunk_tids[0]}"

            # How much of this chunk can be mapped in one go?
            mismatches = (chunk_tids != target_tids).nonzero()[0]
            if mismatches.size > 0:
                n_match = mismatches[0]
            else:
                n_match = len(chunk_tids)

            # Select the matching data
            chunk_match_end = chunk_match_start + n_match
            tgt_end = tgt_start + n_match

            yield slice(tgt_start, tgt_end), slice(chunk_match_start, chunk_match_end)

            # Prepare remaining data in the chunk for the next match
            chunk_match_start = chunk_match_end
            chunk_tids = chunk_tids[n_match:]

    @property
    def train_ids(self):
        return self.data.train_ids

    @property
    def frames_per_train(self):
        counts = set(self.frame_counts.unique()) - {0}
        if len(counts) > 1:
            raise ValueError(f"Varying number of frames per train: {counts}")
        return counts.pop()

    def __repr__(self):
        return "<{}: Data interface for detector {!r} with {} modules>".format(
            type(self).__name__, self.detector_name, len(self.source_to_modno),
        )

    @staticmethod
    def _concat(arrays, index, fill_value, astype):
        dtype = arrays[0].dtype if astype is None else np.dtype(astype)
        if fill_value is None:
            fill_value = np.nan if dtype.kind == 'f' else 0
        fill_value = dtype.type(fill_value)

        return xarray.concat(
            [a.astype(dtype, copy=False) for a in arrays],
            pd.Index(index, name='module'),
            fill_value=fill_value
        )

    @staticmethod
    def _out_array(shape, dtype, fill_value=None):
        if fill_value is None:
            fill_value = np.nan if dtype.kind == 'f' else 0
        fill_value = dtype.type(fill_value)

        if fill_value == 0:
            return np.zeros(shape, dtype=dtype)
        else:
            return np.full(shape, fill_value, dtype=dtype)

    def get_array(self, key, *, fill_value=None, roi=(), astype=None):
        """Get a labelled array of detector data

        Parameters
        ----------
        key: str
          The data to get, e.g. 'image.data' for pixel values.
        fill_value: int or float, optional
            Value to use for missing values. If None (default) the fill value
            is 0 for integers and np.nan for floats.
        roi: tuple
          Specify e.g. ``np.s_[10:60, 100:200]`` to select pixels within each
          module when reading data. The selection is applied to each individual
          module, so it may only be useful when working with a single module.
        astype: Type
          Data type of the output array. If None (default) the dtype matches the
          input array dtype
        """
        train_ids = np.asarray(self.data.train_ids)

        eg_src = min(self.source_to_modno)
        eg_keydata = self.data[eg_src, key]

        # Find the shape of 1 frame for 1 module with the ROI applied
        out_shape = ((len(self.modno_to_source), len(train_ids))
                     + roi_shape(eg_keydata.entry_shape, roi))

        dtype = eg_keydata.dtype if astype is None else np.dtype(astype)
        out = self._out_array(out_shape, dtype, fill_value=fill_value)


        modnos = []
        for mod_ix, (modno, source) in enumerate(sorted(self.modno_to_source.items())):
            for chunk in self.data._find_data_chunks(source, key):
                for tgt_slice, chunk_slice in self._split_align_chunk(chunk, train_ids):
                    chunk.dataset.read_direct(
                        out[mod_ix, tgt_slice], source_sel=(chunk_slice,) + roi
                    )

            modnos.append(modno)

        # Dimension labels
        dims = ['module', 'trainId'] + ['dim_%d' % i for i in range(out.ndim - 2)]

        # Train ID index
        coords = {'module': modnos, 'trainId': train_ids}

        return xarray.DataArray(out, dims=dims, coords=coords)


    def get_dask_array(self, key, fill_value=None, astype=None):
        """Get a labelled Dask array of detector data

        Parameters
        ----------
        key: str
          The data to get, e.g. 'image.data' for pixel values.
        fill_value: int or float, optional
          Value to use for missing values. If None (default) the fill value is 0
          for integers and np.nan for floats.
        astype: Type
          Data type of the output array. If None (default) the dtype matches the
          input array dtype
        """
        arrays = []
        modnos = []
        for modno, source in sorted(self.modno_to_source.items()):
            modnos.append(modno)
            mod_arr = self.data.get_dask_array(source, key, labelled=True)
            arrays.append(mod_arr)

        return self._concat(arrays, modnos, fill_value, astype)

    def trains(self, require_all=True):
        """Iterate over trains for detector data.

        Parameters
        ----------
        require_all: bool
          If True (default), skip trains where any of the selected detector
          modules are missing data.

        Yields
        ------
        train_data: dict
          A dictionary mapping key names (e.g. ``image.data``) to labelled
          arrays.
        """
        return MPxDetectorTrainIterator(self, require_all=require_all)


class XtdfDetectorBase(MultimodDetectorBase):
    """Common machinery for a group of detectors with similar data format

    AGIPD, DSSC & LPD all store pulse-resolved data in an "image" group,
    with both trains and pulses along the first dimension. This allows a
    different number of frames to be stored for each train, which makes
    access more complicated.
    """
    n_modules = 16
    _main_data_key = 'image.data'

    def __init__(self, data: DataCollection, detector_name=None, modules=None,
                 *, min_modules=1):
        super().__init__(data, detector_name, modules, min_modules=min_modules)

    @staticmethod
    def _select_pulse_ids(pulses, data_pulse_ids):
        """Select pulses by ID across a chunk of trains

        Returns an array or slice of the indexes to include.
        """
        if isinstance(pulses.value, slice):
            s = pulses.value
            desired = np.arange(s.start, s.stop, step=s.step, dtype=np.uint64)
        else:
            desired = pulses.value

        return np.isin(data_pulse_ids, desired)

    @staticmethod
    def _select_pulse_indices(pulses, counts):
        """Select pulses by index across a chunk of trains

        Returns a boolean array of frames to include.
        """
        sel_frames = np.zeros(counts.sum(), dtype=np.bool_)
        cursor = 0
        for count in counts:
            sel_in_train = pulses.value
            if isinstance(sel_in_train, np.ndarray):
                # Ignore any indices after the end of the train
                sel_in_train = sel_in_train[sel_in_train < count]
            sel_frames[cursor:cursor + count][sel_in_train] = 1
            cursor += count

        return sel_frames

    def _make_image_index(self, tids, inner_ids, inner_name='pulse'):
        """
        Prepare indices for data per inner coordinate.

        Parameters
        ----------
        tids: np.array
          Train id repeated for each inner coordinate.
        inner_ids: np.array
          Array of inner coordinate values.
        inner_name: string
          Name of the inner coordinate.

        Returns
        -------
        pd.MultiIndex
          MultiIndex of 'train_ids' x 'inner_ids'.
        """
        # Overridden in LPD1M for parallel gain mode
        return pd.MultiIndex.from_arrays(
            [tids, inner_ids], names=['train', inner_name]
        )

    @staticmethod
    def _guess_axes(data, train_pulse_ids, unstack_pulses, modnos=None):
        shape = data.shape
        if modnos is not None:
            shape = shape[1:]
        ndim = len(shape)

        # Raw files have a spurious extra dimension
        if ndim >= 2 and shape[1] == 1:
            if modnos is None:
                data = data[:, 0]
            else:
                data = data[:, :, 0]
            ndim -= 1

        # TODO: this assumes we can tell what the axes are just from the
        # number of dimensions. Works for the data we've seen, but we
        # should look for a more reliable way.
        if ndim == 4:
            # image.data in raw data
            dims = ['train_pulse', 'data_gain', 'slow_scan', 'fast_scan']
        elif ndim == 3:
            # image.data, image.gain, image.mask in calibrated data
            dims = ['train_pulse', 'slow_scan', 'fast_scan']
        else:
            # Everything else seems to be 1D
            dims = ['train_pulse']

        coords = {'train_pulse': train_pulse_ids}
        if modnos is not None:
            dims = ['module'] + dims
            coords['module'] = modnos

        arr = xarray.DataArray(data, coords=coords, dims=dims)

        if unstack_pulses:
            # Separate train & pulse dimensions, and arrange dimensions
            # so that the data is contiguous in memory.
            if modnos is None:
                dim_order = train_pulse_ids.names + dims[1:]
            else:
                dim_order = ['module'] + train_pulse_ids.names + dims[2:]
            return arr.unstack('train_pulse').transpose(*dim_order)
        else:
            return arr

    def _read_inner_ids(self, field='pulseId'):
        """Read pulse/cell IDs into a 2D array (frames, modules)

        Overridden by LPD1M for parallel gain mode.
        """
        inner_ids = np.full((
            self.frame_counts.sum(), self.n_modules), NO_PULSE_ID, dtype=np.uint64
        )

        for source, modno in self.source_to_modno.items():
            for chunk in self.data._find_data_chunks(source, 'image.' + field):
                dset = chunk.dataset
                unwanted_dim = (dset.ndim > 1)  and (dset.shape[1] == 1)
                for tgt_slice, chunk_slice in self._split_align_chunk(
                        chunk, self.train_ids_perframe
                ):
                    # Select the matching data and add it to pulse_ids
                    # In some cases, there's an extra dimension of length 1.
                    matched = chunk.dataset[chunk_slice]
                    if unwanted_dim:
                        matched = matched[:, 0]
                    inner_ids[tgt_slice, modno] = matched

        return inner_ids

    def _collect_inner_ids(self, field='pulseId'):
        """
        Gather pulse/cell ID labels for all modules and check consistency.

        Raises
        ------
        Exception:
          Some data has no pulse ID values for any module.
        Exception:
          Inconsistent pulse IDs between detector modules.

        Returns
        -------
        inner_ids: np.array
          Array of pulse/cell IDs per frame common for all detector modules.
        """
        inner_ids = self._read_inner_ids(field)
        # Sanity checks on pulse IDs
        inner_ids_min: np.ndarray = inner_ids.min(axis=1)
        if (inner_ids_min == NO_PULSE_ID).any():
            raise Exception(f"Failed to find {field} for some data")
        inner_ids[inner_ids == NO_PULSE_ID] = 0
        if (inner_ids_min != inner_ids.max(axis=1)).any():
            raise Exception(f"Inconsistent {field} for different modules")

        # Pulse IDs make sense. Drop the modules dimension, giving one
        # pulse ID for each frame.
        return inner_ids_min

    def _read_chunk(self, chunk: DataChunk, sel_frames, mod_out, roi):
        """Read per-pulse data from file into an output array (of 1 module)"""
        for tgt_slice, chunk_slice in self._split_align_chunk(
                chunk, self.train_ids_perframe
        ):
            inc_pulses_chunk = sel_frames[tgt_slice]
            if inc_pulses_chunk.sum() == 0:  # No data from this chunk selected
                continue
            elif inc_pulses_chunk.all():  # All pulses in chunk
                chunk.dataset.read_direct(
                    mod_out[tgt_slice], source_sel=(chunk_slice,) + roi
                )
                continue

            # Read a subset of pulses from the chunk:

            # Reading a non-contiguous selection in HDF5 seems to be slow:
            # https://forum.hdfgroup.org/t/performance-reading-data-with-non-contiguous-selection/8979
            # Except it's fast if you read the data to a matching selection in
            # memory (one weird trick).
            # So as a workaround, this allocates a temporary array of the same
            # shape as the dataset, reads into it, and then copies the selected
            # data to the output array. The extra memory copy is not optimal,
            # but it's better than the HDF5 performance issue, at least in some
            # realistic cases.
            # N.B. tmp should only use memory for the data it contains -
            # zeros() uses calloc, so the OS can do virtual memory tricks.
            # Don't change this to zeros_like() !
            tmp = np.zeros(chunk.dataset.shape, chunk.dataset.dtype)
            pulse_sel = np.nonzero(inc_pulses_chunk)[0] + chunk_slice.start
            sel_region = (pulse_sel,) + roi
            chunk.dataset.read_direct(
                tmp, source_sel=sel_region, dest_sel=sel_region,
            )
            # Where does this data go in the target array?
            tgt_start_ix = sel_frames[:tgt_slice.start].sum()
            tgt_pulse_sel = slice(
                tgt_start_ix, tgt_start_ix + inc_pulses_chunk.sum()
            )
            # Copy data from temp array to output array
            tmp_frames_mask = np.zeros(len(tmp), dtype=np.bool_)
            tmp_frames_mask[pulse_sel] = True
            np.compress(
                tmp_frames_mask, tmp[np.index_exp[:] + roi],
                axis=0, out=mod_out[tgt_pulse_sel]
            )

    def _get_pulse_data(self, key, pulses, unstack_pulses=True,
                        fill_value=None, subtrain_index='pulseId', roi=(),
                        astype=None):
        """Get a labelled array of per-pulse data (image.*) for xtdf detector"""
        pulses = _check_pulse_selection(pulses)

        if isinstance(pulses, by_index):
            sel_frames = self._select_pulse_indices(pulses, self.frame_counts)
            pulse_ids = None
        else:  # by_id
            pulse_ids = self._collect_inner_ids('pulseId')
            sel_frames = self._select_pulse_ids(pulses, pulse_ids)

        nframes_sel = sel_frames.sum()

        eg_src = min(self.source_to_modno)
        eg_keydata = self.data[eg_src, key]

        if eg_keydata.ndim >= 2 and eg_keydata.entry_shape[0] == 1:
            # Ensure ROI applies to pixel dimensions, not the extra
            # dim in raw data (except AGIPD, where it is data/gain)
            roi = np.index_exp[:] + roi

        _roi_shape = roi_shape(eg_keydata.entry_shape, roi)
        out_shape = (len(self.modno_to_source), nframes_sel) + _roi_shape

        dtype = eg_keydata.dtype if astype is None else np.dtype(astype)
        out = self._out_array(out_shape, dtype, fill_value=fill_value)

        modnos = []
        for mod_ix, (modno, source) in enumerate(sorted(self.modno_to_source.items())):
            for chunk in self.data._find_data_chunks(source, key):
                self._read_chunk(chunk, sel_frames, out[mod_ix], roi)

            modnos.append(modno)
        
        if (subtrain_index == 'pulseId') and (pulse_ids is not None):
            inner_ids = pulse_ids
        else:
            inner_ids = self._collect_inner_ids(subtrain_index)

        index = self._make_image_index(
            self.train_ids_perframe, inner_ids, subtrain_index[:-2]
        )[sel_frames]
        
        return self._guess_axes(out, index, unstack_pulses, modnos=modnos)

    def get_array(self, key, pulses=np.s_[:], unstack_pulses=True, *,
                  fill_value=None, subtrain_index='pulseId', roi=(),
                  astype=None):
        """Get a labelled array of detector data

        Parameters
        ----------
        key: str
          The data to get, e.g. 'image.data' for pixel values.
        pulses: slice, array, by_id or by_index
          Select the pulses to include from each train. by_id selects by pulse
          ID, by_index by index within the data being read. The default includes
          all pulses. Only used for per-pulse data.
        unstack_pulses: bool
          Whether to separate train and pulse dimensions.
        fill_value: int or float, optional
          Value to use for missing values. If None (default) the fill value is 0
          for integers and np.nan for floats.
        subtrain_index: str
          Specify 'pulseId' (default) or 'cellId' to label the frames recorded
          within each train. Pulse ID should allow this data to be matched with
          other devices, but depends on how the detector was manually configured
          when the data was taken. Cell ID refers to the memory cell used for
          that frame in the detector hardware.
        roi: tuple
          Specify e.g. ``np.s_[10:60, 100:200]`` to select pixels within each
          module when reading data. The selection is applied to each individual
          module, so it may only be useful when working with a single module.
          For AGIPD raw data, each module records a frame as a 3D array with 2
          entries on the first dimension, for data & gain information, so
          ``roi=np.s_[0]`` will select only the data part of each frame.
        astype: Type
          data type of the output array. If None (default) the dtype matches the
          input array dtype
        """
        if subtrain_index not in {'pulseId', 'cellId'}:
            raise ValueError("subtrain_index must be 'pulseId' or 'cellId'")
        if not isinstance(roi, tuple):
            roi = (roi,)

        if key.startswith('image.'):
            return self._get_pulse_data(
                key, pulses, unstack_pulses, fill_value=fill_value,
                subtrain_index=subtrain_index, roi=roi, astype=astype,
            )
        else:
            return super().get_array(
                key, fill_value=fill_value, roi=roi, astype=astype
            )

    def get_dask_array(self, key, subtrain_index='pulseId', fill_value=None,
                       astype=None):
        """Get a labelled Dask array of detector data

        Dask does lazy, parallelised computing, and can work with large data
        volumes. This method doesn't immediately load the data: that only
        happens once you trigger a computation.

        Parameters
        ----------
        key: str
          The data to get, e.g. 'image.data' for pixel values.
        subtrain_index: str, optional
          Specify 'pulseId' (default) or 'cellId' to label the frames recorded
          within each train. Pulse ID should allow this data to be matched with
          other devices, but depends on how the detector was manually configured
          when the data was taken. Cell ID refers to the memory cell used for
          that frame in the detector hardware.
        fill_value: int or float, optional
          Value to use for missing values. If None (default) the fill value is 0
          for integers and np.nan for floats.
        astype: Type, optional
          data type of the output array. If None (default) the dtype matches the
          input array dtype
        """
        if subtrain_index not in {'pulseId', 'cellId'}:
            raise ValueError("subtrain_index must be 'pulseId' or 'cellId'")
        arrays = []
        modnos = []
        for modno, source in sorted(self.modno_to_source.items()):
            modnos.append(modno)
            mod_arr = self.data.get_dask_array(source, key, labelled=True)

            # At present, all the per-pulse data is stored in the 'image' key.
            # If that changes, this check will need to change as well.
            if key.startswith('image.'):
                # Add pulse IDs to create multi-level index
                inner_ix = self.data.get_array(source, 'image.' + subtrain_index)
                # Raw files have a spurious extra dimension
                if inner_ix.ndim >= 2 and inner_ix.shape[1] == 1:
                    inner_ix = inner_ix[:, 0]

                mod_arr = mod_arr.rename({'trainId': 'train_pulse'})

                mod_arr.coords['train_pulse'] = self._make_image_index(
                    mod_arr.coords['train_pulse'].values, inner_ix.values,
                    inner_name=subtrain_index,
                ).set_names('trainId', level=0)
                # This uses 'trainId' where a concrete array from the same class
                # uses 'train'. I didn't notice that inconsistency when I
                # introduced it, and now code may be relying on each name.

            arrays.append(mod_arr)

        return self._concat(arrays, modnos, fill_value, astype)

    def trains(self, pulses=np.s_[:], require_all=True):
        """Iterate over trains for detector data.

        Parameters
        ----------
        pulses: slice, array, by_index or by_id
          Select which pulses to include for each train.
          The default is to include all pulses.
        require_all: bool
          If True (default), skip trains where any of the selected detector
          modules are missing data.

        Yields
        ------
        train_data: dict
          A dictionary mapping key names (e.g. ``image.data``) to labelled
          arrays.
        """
        return MPxDetectorTrainIterator(self, pulses, require_all=require_all)

    def write_virtual_cxi(self, filename, fillvalues=None):
        """Write a virtual CXI file to access the detector data.

        The virtual datasets in the file provide a view of the detector
        data as if it was a single huge array, but without copying the data.
        Creating and using virtual datasets requires HDF5 1.10.

        Parameters
        ----------
        filename: str
          The file to be written. Will be overwritten if it already exists.
        fillvalues: dict, optional
            keys are datasets names (one of: data, gain, mask) and associated
            fill value for missing data  (default is np.nan for float arrays and
            zero for integer arrays)
        """
        XtdfCXIWriter(self).write(filename, fillvalues=fillvalues)

    def write_frames(self, filename, trains, pulses):
        """Write selected detector frames to a new EuXFEL HDF5 file

        trains and pulses should be 1D arrays of the same length, containing
        train IDs and pulse IDs (corresponding to the pulse IDs recorded by
        the detector). i.e. (trains[i], pulses[i]) identifies one frame.
        """
        if (trains.ndim != 1) or (pulses.ndim != 1):
            raise ValueError("trains & pulses must be 1D arrays")
        inc_tp_ids = zip_trains_pulses(trains, pulses)

        writer = FramesFileWriter(filename, self.data, inc_tp_ids)
        try:
            writer.write()
        finally:
            writer.file.close()


def zip_trains_pulses(trains, pulses):
    """Combine two similar arrays of train & pulse IDs as one struct array
    """
    if trains.shape != pulses.shape:
        raise ValueError(
            f"Train & pulse arrays don't match ({trains.shape} != {pulses.shape})"
        )

    res = np.zeros(trains.shape, dtype=np.dtype([
        ('trainId', np.uint64), ('pulseId', np.uint64)
    ]))
    res['trainId'] = trains
    res['pulseId'] = pulses
    return res


class FramesFileWriter(FileWriter):
    """Write selected detector frames in European XFEL HDF5 format"""
    def __init__(self, path, data, inc_tp_ids):
        super().__init__(path, data)
        self.inc_tp_ids = inc_tp_ids


    def _guess_number_of_storing_entries(self, source, key):
        if source in self.data.instrument_sources and key.startswith("image."):
            # Start with an empty dataset, grow it as we add each file
            return 0
        else:
            return super()._guess_number_of_storing_entries(source, key)

    def copy_image_data(self, source, keys):
        """Copy selected frames of the detector image data"""
        frame_tids_piecewise = []

        src_files = sorted(
            self.data._source_index[source],
            key=lambda fa: fa.train_ids[0]
        )

        for fa in src_files:
            _, counts = fa.get_index(source, 'image')
            file_tids = np.repeat(fa.train_ids, counts.astype(np.intp))
            file_pids = fa.file[f'/INSTRUMENT/{source}/image/pulseId'][:]
            if file_pids.ndim == 2 and file_pids.shape[1] == 1:
                # Raw data has a spurious extra dimension
                file_pids = file_pids[:, 0]

            # Data can have trailing 0s, seemingly
            file_pids = file_pids[:len(file_tids)]
            file_tp_ids = zip_trains_pulses(file_tids, file_pids)

            # indexes of selected frames in datasets under .../image in this file
            ixs = np.isin(file_tp_ids, self.inc_tp_ids).nonzero()[0]
            nframes = ixs.shape[0]

            for key in keys:
                path = f"INSTRUMENT/{source}/{key.replace('.', '/')}"

                dst_ds = self.file[path]
                dst_cursor = dst_ds.shape[0]
                dst_ds.resize(dst_cursor + nframes, axis=0)
                dst_ds[dst_cursor: dst_cursor+nframes] = fa.file[path][ixs]

            frame_tids_piecewise.append(file_tids[ixs])

        frame_tids = np.concatenate(frame_tids_piecewise)
        self._make_index(source, 'image', frame_tids)

    def copy_source(self, source):
        """Copy all the relevant data for one detector source"""
        if source not in self.data.instrument_sources:
            return super().copy_source(source)

        all_keys = self.data.keys_for_source(source)
        img_keys = {k for k in all_keys if k.startswith('image.')}

        for key in sorted(all_keys - img_keys):
            self.copy_dataset(source, key)

        self.copy_image_data(source, sorted(img_keys))


class MPxDetectorTrainIterator:
    """Iterate over trains in detector data, assembling arrays.

    Created by :meth:`DetectorData.trains`.
    """
    def __init__(self, data, pulses=by_index[:], require_all=True):
        self.data = data
        self.pulses = _check_pulse_selection(pulses)
        self.require_all = require_all
        # {(source, key): (f, dataset)}
        self._datasets_cache = {}

    def _find_data(self, source, key, tid):
        """
        Find FileAccess instance and dataset corresponding to source, key,
        and train id tid.

        Parameters
        ----------
        source: string
          Path to keys in HD5 file, e.g.: 'SPB_DET_AGIPD1M-1/DET/5CH0:xtdf'.
        key: string
          Key for data at source separated by dot, e.g.: 'image.data'.
        tid: np.int
          Train id.

        Returns
        -------
        Tuple[FileAccess, int, h5py.Dataset]
          FileAccess
            Instance for the HD5 file with requested data.
          int
            Starting index for the requested data.
          h5py.Dataset
            h5py dataset with found data.
        """
        file, ds = self._datasets_cache.get((source, key), (None, None))
        if ds:
            ixs = (file.train_ids == tid).nonzero()[0]
            if ixs.size > 0:
                return file, ixs[0], ds

        data = self.data.data
        path = '/INSTRUMENT/{}/{}'.format(source, key.replace('.', '/'))
        f, pos = data._find_data(source, tid)
        if f is not None:
            ds = f.file[path]
            self._datasets_cache[(source, key)] = (f, ds)
            return f, pos, ds

        return None, None, None

    def _get_slow_data(self, source, key, tid):
        """
        Get an array of slow (per train) data corresponding to source, key,
        and train id tid. Also used for JUNGFRAU data with memory cell
        dimension.

        Parameters
        ----------
        source: string
          Path to keys in HD5 file, e.g.: 'SPB_DET_AGIPD1M-1/DET/5CH0:xtdf'.
        key: string
          Key for data at source separated by dot, e.g.: 'header.pulseCount'.
        tid: np.int
          Train id.

        Returns
        -------
        xarray.DataArray
          Array of selected slow data. In case there are more than one frame
          for the train id tid - train id dimension is kept indexing frames
          within tid.
        """
        file, pos, ds = self._find_data(source, key, tid)
        if file is None:
            return None

        group = key.partition('.')[0]
        firsts, counts = file.get_index(source, group)
        first, count = firsts[pos], counts[pos]
        if count == 1:
            return xarray.DataArray(ds[first])
        else:
            return xarray.DataArray(ds[first : first + count])

    def _get_pulse_data(self, source, key, tid):
        """
        Get an array of per pulse data corresponding to source, key,
        and train id tid. Used only for AGIPD-like detectors, for 
        JUNGFRAU-like per-cell data '_get_slow_data' is used.

        Parameters
        ----------
        source: string
          Path to keys in HD5 file, e.g.: 'SPB_DET_AGIPD1M-1/DET/5CH0:xtdf'.
        key: string
          Key for data at source separated by dot, e.g.: 'image.data'.
        tid: np.int
          Train id.

        Returns
        -------
        xarray.DataArray
          Array of selected per pulse data.
        """
        file, pos, ds = self._find_data(source, key, tid)
        if file is None:
            return None

        group = key.partition('.')[0]
        firsts, counts = file.get_index(source, group)
        first, count = firsts[pos], counts[pos]

        pulse_ids = file.file['/INSTRUMENT/{}/{}/pulseId'.format(source, group)][
            first : first + count
        ]
        # Raw files have a spurious extra dimension
        if pulse_ids.ndim >= 2 and pulse_ids.shape[1] == 1:
            pulse_ids = pulse_ids[:, 0]

        if isinstance(self.pulses, by_id):
            positions = self._select_pulse_ids(pulse_ids)
        else:  # by_index
            positions = self._select_pulse_indices(count)
        pulse_ids = pulse_ids[positions]
        train_ids = np.array([tid] * len(pulse_ids), dtype=np.uint64)
        train_pulse_ids = self.data._make_image_index(train_ids, pulse_ids)

        if isinstance(positions, slice):
            data_positions = slice(
                int(first + positions.start),
                int(first + positions.stop),
                positions.step
            )
        else:  # ndarray
            data_positions = first + positions

        return self.data._guess_axes(
            ds[data_positions], train_pulse_ids, unstack_pulses=True
        )

    def _select_pulse_ids(self, pulse_ids):
        """Select pulses by ID

        Returns an array or slice of the indexes to include.
        """
        val = self.pulses.value
        N = len(pulse_ids)
        if isinstance(val, slice):
            if val.step == 1:
                after_start = np.nonzero(pulse_ids >= val.start)[0]
                after_stop = np.nonzero(pulse_ids >= val.stop)[0]
                start_ix = after_start[0] if (after_start.size > 0) else N
                stop_ix = after_stop[0] if (after_stop.size > 0) else N
                return slice(start_ix, stop_ix)

            # step != 1
            desired = np.arange(val.start, val.stop, step=val.step, dtype=np.uint64)

        else:
            desired = val

        return np.nonzero(np.isin(pulse_ids, desired))[0]

    def _select_pulse_indices(self, count):
        """Select pulses by index

        Returns an array or slice of the indexes to include.
        """
        val = self.pulses.value
        if isinstance(val, slice):
            return slice(val.start, min(val.stop, count), val.step)

        # ndarray
        return val[val < count]

    def _assemble_data(self, tid):
        """
        Assemble data for all keys into a dictionary for specified train id.

        Parameters
        ----------
        tid: int
          Train id.

        Returns
        -------
        Dict[str, xarray]:
          str
            Key name.
          xarray
            Assembled data array.
        """
        key_module_arrays = {}

        for modno, source in sorted(self.data.modno_to_source.items()):

            for key in self.data.data._keys_for_source(source):
                # At present, all the per-pulse data is stored in the 'image' key.
                # If that changes, this check will need to change as well.

                if key.startswith('image.'):
                    mod_data = self._get_pulse_data(source, key, tid)
                else:
                    mod_data = self._get_slow_data(source, key, tid)

                if mod_data is None:
                    continue

                if key not in key_module_arrays:
                    key_module_arrays[key] = [], []
                modnos, data = key_module_arrays[key]
                modnos.append(modno)
                data.append(mod_data)

        # Assemble the data for each key into one xarray
        return {
            k: xarray.concat(data, pd.Index(modnos, name='module'))
            for (k, (modnos, data)) in key_module_arrays.items()
        }

    def __iter__(self):
        """
        Iterate over train ids and yield assembled data dictionaries.

        Yields
        ------
        Tuple[int, Dict[str, xarray]]:
          int
            train id.
          Dict[str, xarray]
            assembled {key: data array} dictionary.
        """
        for tid in self.data.train_ids:
            tid = int(tid)  # Convert numpy int to regular Python int
            if self.require_all and self.data.data._check_data_missing(tid):
                continue
            yield tid, self._assemble_data(tid)


@multimod_detectors
class AGIPD1M(XtdfDetectorBase):
    """An interface to AGIPD-1M data.

    Parameters
    ----------
    data: DataCollection
      A data collection, e.g. from :func:`.RunDirectory`.
    modules: set of ints, optional
      Detector module numbers to use. By default, all available modules
      are used.
    detector_name: str, optional
      Name of a detector, e.g. 'SPB_DET_AGIPD1M-1'. This is only needed
      if the dataset includes more than one AGIPD detector.
    min_modules: int
      Include trains where at least n modules have data. Default is 1.
    """
    _source_re = re.compile(r'(?P<detname>.+_AGIPD1M.*)/DET/(?P<modno>\d+)CH')
    module_shape = (512, 128)


@multimod_detectors
class AGIPD500K(XtdfDetectorBase):
    """An interface to AGIPD-500K data

    Detector names are like 'HED_DET_AGIPD500K2G', otherwise this is identical
    to :class:`AGIPD1M`.
    """
    _source_re = re.compile(r'(?P<detname>.+_AGIPD500K.*)/DET/(?P<modno>\d+)CH')
    module_shape = (512, 128)
    n_modules = 8


@multimod_detectors
class DSSC1M(XtdfDetectorBase):
    """An interface to DSSC-1M data.

    Parameters
    ----------
    data: DataCollection
      A data collection, e.g. from :func:`.RunDirectory`.
    modules: set of ints, optional
      Detector module numbers to use. By default, all available modules
      are used.
    detector_name: str, optional
      Name of a detector, e.g. 'SCS_DET_DSSC1M-1'. This is only needed
      if the dataset includes more than one DSSC detector.
    min_modules: int
      Include trains where at least n modules have data. Default is 1.
    """
    _source_re = re.compile(r'(?P<detname>.+_DSSC1M.*)/DET/(?P<modno>\d+)CH')
    module_shape = (128, 512)


@multimod_detectors
class LPD1M(XtdfDetectorBase):
    """An interface to LPD-1M data.

    Parameters
    ----------
    data: DataCollection
      A data collection, e.g. from :func:`.RunDirectory`.
    modules: set of ints, optional
      Detector module numbers to use. By default, all available modules
      are used.
    detector_name: str, optional
      Name of a detector, e.g. 'FXE_DET_LPD1M-1'. This is only needed
      if the dataset includes more than one LPD detector.
    min_modules: int
      Include trains where at least n modules have data. Default is 1.
    parallel_gain: bool
      Set to True to read this data as parallel gain data, where high, medium
      and low gain data are stored sequentially within each train. This will
      repeat the pulse & cell IDs from the first 1/3 of each train, and add gain
      stage labels from 0 (high-gain) to 2 (low-gain).
    """
    _source_re = re.compile(r'(?P<detname>.+_LPD1M.*)/DET/(?P<modno>\d+)CH')
    module_shape = (256, 256)

    def __init__(self, data: DataCollection, detector_name=None, modules=None,
                 *, min_modules=1, parallel_gain=False):
        super().__init__(data, detector_name, modules, min_modules=min_modules)

        self.parallel_gain = parallel_gain
        if parallel_gain:
            if ((self.frame_counts % 3) != 0).any():
                raise ValueError(
                    "parallel_gain=True needs the frames in each train to be divisible by 3"
                )

    def _read_inner_ids(self, field='pulseId'):
        inner_ids = super()._read_inner_ids(field)

        if not self.parallel_gain:
            return inner_ids

        # In 'parallel gain' mode, the first 1/3 of pulse/cell IDs in each train
        # are valid, but the remaining 2/3 are junk. So we'll repeat the valid
        # ones 3 times (in inner_ids_fixed).
        inner_ids_fixed = np.zeros_like(inner_ids)

        cursor = 0
        for count in self.frame_counts:  # Iterate through trains
            n_per_gain_stage = int(count // 3)
            train_inner_ids = inner_ids[cursor: cursor + n_per_gain_stage]
            for stage in range(3):
                end = cursor + n_per_gain_stage
                inner_ids_fixed[cursor:end] = train_inner_ids
                cursor = end
        return inner_ids_fixed

    def _select_pulse_indices(self, pulses, counts):
        """Select pulses by index across a chunk of trains

        Returns a boolean array of frames to include.
        """
        if not self.parallel_gain:
            return super()._select_pulse_indices(pulses, counts)

        sel_frames = np.zeros(counts.sum(), dtype=np.bool_)
        cursor = 0
        for count in counts:
            n_per_gain_stage = int(count // 3)
            sel_in_train = pulses.value
            if isinstance(sel_in_train, np.ndarray):
                # Ignore any indices after the end of the gain stage
                sel_in_train = sel_in_train[sel_in_train < n_per_gain_stage]

            for stage in range(3):
                sel_frames[cursor:cursor + n_per_gain_stage][sel_in_train] = 1
                cursor += n_per_gain_stage

        return sel_frames

    def _make_image_index(self, tids, inner_ids, inner_name='pulse'):
        if not self.parallel_gain:
            return super()._make_image_index(tids, inner_ids, inner_name)

        # In 'parallel gain' mode, the first 1/3 of pulse/cell IDs in each train
        # are valid, but the remaining 2/3 are junk. So we'll repeat the valid
        # ones 3 times (in inner_ids_fixed). At the same time, we make a gain
        # stage index (0-2), so each frame has a unique entry in the MultiIndex
        # (train ID, gain, pulse/cell ID)
        gain = np.zeros_like(inner_ids, dtype=np.uint8)
        inner_ids_fixed = np.zeros_like(inner_ids)

        _, firsts, counts = np.unique(tids, return_index=True, return_counts=True)
        for ix, frames in zip(firsts, counts):  # Iterate through trains
            n_per_gain_stage = int(frames // 3)
            train_inner_ids = inner_ids[ix: ix + n_per_gain_stage]
            for stage in range(3):
                start = ix + (stage * n_per_gain_stage)
                end = start + n_per_gain_stage
                gain[start:end] = stage
                inner_ids_fixed[start:end] = train_inner_ids

        return pd.MultiIndex.from_arrays(
            [tids, gain, inner_ids_fixed], names=['train', 'gain', inner_name]
        )


@multimod_detectors
class JUNGFRAU(MultimodDetectorBase):
    """An interface to JUNGFRAU data.

    JNGFR, JF1M, JF4M all store data in a "data" group, with trains along
    the first and memory cells along the second dimension.
    This allows only a set number of frames to be stored for each train.

    Parameters
    ----------
    data: DataCollection
      A data collection, e.g. from :func:`.RunDirectory`.
    detector_name: str, optional
      Name of a detector, e.g. 'SPB_IRDA_JNGFR'. This is only needed
      if the dataset includes more than one JUNGFRAU detector.
    modules: set of ints, optional
      Detector module numbers to use. By default, all available modules
      are used.
    min_modules: int
      Include trains where at least n modules have data. Default is 1.
    n_modules: int
      Number of detector modules in the experiment setup. Default is
      None, in which case it will be estimated from the available data.
    """
    # We appear to have a few different formats for source names:
    # SPB_IRDA_JNGFR/DET/MODULE_1:daqOutput  (e.g. in p 2566, r 61)
    # SPB_IRDA_JF4M/DET/JNGFR03:daqOutput    (e.g. in p 2732, r 12)
    # FXE_XAD_JF1M/DET/RECEIVER-1
    _source_re = re.compile(
        r'(?P<detname>.+_(JNGFR|JF[14]M))/DET/'
        r'(MODULE_|RECEIVER-|JNGFR)(?P<modno>\d+)'
    )
    _main_data_key = 'data.adc'
    module_shape = (512, 1024)

    def __init__(self, data: DataCollection, detector_name=None, modules=None,
                 *, min_modules=1, n_modules=None):
        super().__init__(data, detector_name, modules, min_modules=min_modules)

        if n_modules is not None:
            self.n_modules = int(n_modules)
        else:
            # For JUNGFRAU modules are indexed from 1
            self.n_modules = max(modno for (_, modno) in self._source_matches(
                data, self.detector_name
            ))

    @staticmethod
    def _label_dims(arr):
        # Label dimensions to match the AGIPD/DSSC/LPD data access
        ndim_pertrain = arr.ndim
        if 'trainId' in arr.dims:
            arr = arr.rename({'trainId': 'train'})
            ndim_pertrain = arr.ndim - 1

        if ndim_pertrain == 4:
            arr = arr.rename({
                'dim_0': 'cell', 'dim_1': 'slow_scan', 'dim_2': 'fast_scan'
            })
        elif ndim_pertrain == 2:
            arr = arr.rename({'dim_0': 'cell'})
        return arr

    def get_array(self, key, *, fill_value=None, roi=(), astype=None):
        """Get a labelled array of detector data

        Parameters
        ----------
        key: str
          The data to get, e.g. 'data.adc' for pixel values.
        fill_value: int or float, optional
            Value to use for missing values. If None (default) the fill value
            is 0 for integers and np.nan for floats.
        roi: tuple
          Specify e.g. ``np.s_[:, 10:60, 100:200]`` to select data within each
          module & each train when reading data. The first dimension is pulses,
          then there are two pixel dimensions. The same selection is applied
          to data from each module, so selecting pixels may only make sense if
          you're using a single module.
        astype: Type
          data type of the output array. If None (default) the dtype matches the
          input array dtype
        """
        arr = super().get_array(key, fill_value=fill_value, roi=roi, astype=astype)
        return self._label_dims(arr)

    def get_dask_array(self, key, fill_value=None, astype=None):
        """Get a labelled Dask array of detector data

        Dask does lazy, parallelised computing, and can work with large data
        volumes. This method doesn't immediately load the data: that only
        happens once you trigger a computation.

        Parameters
        ----------
        key: str
          The data to get, e.g. 'data.adc' for pixel values.
        fill_value: int or float, optional
          Value to use for missing values. If None (default) the fill value
          is 0 for integers and np.nan for floats.
        astype: Type
          data type of the output array. If None (default) the dtype matches the
          input array dtype
        """
        arr = super().get_dask_array(key, fill_value=fill_value, astype=astype)
        return self._label_dims(arr)

    def trains(self, require_all=True):
        """Iterate over trains for detector data.

        Parameters
        ----------
        require_all: bool
          If True (default), skip trains where any of the selected detector
          modules are missing data.

        Yields
        ------
        train_data: dict
          A dictionary mapping key names (e.g. 'data.adc') to labelled
          arrays.
        """
        for tid, d in super().trains(require_all=require_all):
            yield tid, {k: self._label_dims(a) for (k, a) in d.items()}

    def write_virtual_cxi(self, filename, fillvalues=None):
        """Write a virtual CXI file to access the detector data.

        The virtual datasets in the file provide a view of the detector
        data as if it was a single huge array, but without copying the data.
        Creating and using virtual datasets requires HDF5 1.10.

        Parameters
        ----------
        filename: str
          The file to be written. Will be overwritten if it already exists.
        fillvalues: dict, optional
            keys are datasets names (one of: data, gain, mask) and associated
            fill value for missing data  (default is np.nan for float arrays and
            zero for integer arrays)
        """
        JUNGFRAUCXIWriter(self).write(filename, fillvalues=fillvalues)

def identify_multimod_detectors(
        data, detector_name=None, *, single=False, clses=None
):
    """Identify multi-module detectors in the data

    Various detectors record data for individual X-ray pulses within
    trains, and we often want to process whichever detector was used
    in a run. This tries to identify the detector, so a user doesn't
    have to specify it manually.

    If ``single=True``, this returns a tuple of (detector_name, access_class),
    throwing ``ValueError`` if there isn't exactly 1 detector found.
    If ``single=False``, it returns a set of these tuples.

    *clses* may be a list of acceptable detector classes to check.
    """
    if clses is None:
        clses = multimod_detectors.list

    res = set()
    for cls in clses:
        for source in data.instrument_sources:
            m = cls._source_re.match(source)
            if m:
                name = m.group('detname')
                if (detector_name is None) or (name == detector_name):
                    res.add((name, cls))

    if single:
        if len(res) < 1:
            raise ValueError("No detector sources identified in the data")
        elif len(res) > 1:
            raise ValueError("Multiple detectors identified: {}".format(
                ", ".join(name for (name, _) in res)
            ))
        return res.pop()

    return res
