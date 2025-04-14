"""Interfaces to data from specific instruments
"""
import logging
import math
import re
from collections.abc import Iterable
from copy import copy
from warnings import warn

import numpy as np
import pandas as pd

from .exceptions import SourceNameError
from .reader import DataCollection, by_id, by_index
from .read_machinery import DataChunk, roi_shape, split_trains
from .utils import default_num_threads
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


def _select_pulse_ids(pulses, data_pulse_ids):
    """Select pulses by ID across a chunk of trains

    Returns a boolean array of which entries in data_pulse_ids match.
    """
    if isinstance(pulses.value, slice):
        s = pulses.value
        desired = np.arange(s.start, s.stop, step=s.step, dtype=np.uint64)
    else:
        desired = pulses.value

    return np.isin(data_pulse_ids, desired)


def _out_array(shape, dtype, fill_value=None):
    if fill_value is None:
        fill_value = np.nan if dtype.kind == 'f' else 0
    fill_value = dtype.type(fill_value)

    # Zeroed memory can be allocated faster than explicitly writing zeros
    if fill_value == 0:
        return np.zeros(shape, dtype=dtype)
    else:
        return np.full(shape, fill_value, dtype=dtype)


class MultimodDetectorBase:
    """Base class for detectors made of several modules as separate data sources
    """

    _det_name_pat = r'([^/]+)'
    _source_raw_pat = r'/DET/(?P<modno>\d+)CH'
    _source_corr_pat = r'/CORR/(?P<modno>\d+)CH'
    # Override in subclass
    _main_data_key = ''  # Key to use for checking data counts match
    _mask_data_key = ''
    _frames_per_entry = 1  # Override if separate pulse dimension in files
    _modnos_start_at = 0  # Override if module numbers start at 1 (JUNGFRAU)
    module_shape = (0, 0)
    n_modules = 0

    def __init__(self, data: DataCollection, detector_name=None, modules=None,
                 *, min_modules=1, raw=None):
        if detector_name is None:
            detector_name = self._find_detector_name(data)
        if min_modules <= 0:
            raise ValueError("min_modules must be a positive integer, not "
                             f"{min_modules!r}")

        source_to_modno = self._identify_sources(data, detector_name, modules, raw=raw)

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

        self.frame_counts = frame_counts[self.data.train_ids]

        self.train_ids_perframe = np.repeat(
            self.frame_counts.index.values, self.frame_counts.values.astype(np.intp)
        )
        # If we add extra instance attributes, check whether they should be
        # updated in .select_trains() below.

    def __getitem__(self, item):
        return MultimodKeyData(self, item)

    def __contains__(self, item):
        return all(item in self.data[s] for s in self.source_to_modno)

    def masked_data(self, key=None, *, mask_bits=None, masked_value=np.nan):
        """Combine corrected data with the mask in the files

        This provides an interface similar to ``det['data.adc']``, but masking
        out pixels with the mask from the correction pipeline.

        Parameters
        ----------

        key: str
          The data key to look at, by default the main data key of the detector
          (e.g. 'data.adc').
        mask_bits: int or list of ints
          Reasons to exclude pixels, as a bitmask or a list of integers.
          By default, all types of bad pixel are masked out. See the possible
          values at: https://extra.readthedocs.io/en/latest/calibration/#extra.calibration.BadPixels
        masked_value: int, float
          The replacement value to use for masked data. By default this is NaN.
        """
        key = key or self._main_data_key
        if self._mask_data_key not in self:
            raise RuntimeError(
                f"This data doesn't include a mask ({self._mask_data_key}). "
                f"You might be using raw instead of corrected data."
            )
        if isinstance(mask_bits, Iterable):
            mask_bits = self._combine_bitfield(mask_bits)
        return DetectorMaskedKeyData(
            self, key, mask_key=self._mask_data_key,
            mask_bits=mask_bits, masked_value=masked_value
        )

    @staticmethod
    def _combine_bitfield(ints):
        res = 0
        for i in ints:
            res |= i
        return res

    @classmethod
    def _find_detector_names(cls, data):
        # Find sources matching the pattern (raw or proc) for this detector type
        raw_re  = re.compile(f'(?P<detname>{cls._det_name_pat}){cls._source_raw_pat}')
        corr_re = re.compile(f'(?P<detname>{cls._det_name_pat}){cls._source_corr_pat}')
        detector_names = set()
        for source in data.instrument_sources:
            if m := raw_re.match(source) or corr_re.match(source):
                detector_names.add(m['detname'])
        return detector_names

    @classmethod
    def _find_detector_name(cls, data):
        detector_names = cls._find_detector_names(data)
        # We want exactly 1 source
        if not detector_names:
            raise SourceNameError(f'{cls._det_name_pat}({cls._source_raw_pat}|{cls._source_corr_pat})')
        elif len(detector_names) > 1:
            names_s = ', '.join(repr(n) for n in sorted(detector_names))
            raise ValueError(
                f"Multiple detectors found in the data: {names_s}. "
                f"Pass detector_name to {cls.__name__}() to pick one."
            )
        return detector_names.pop()

    @staticmethod
    def _source_matches(data, pat):
        source_re = re.compile(pat)
        for source in data.instrument_sources:
            m = source_re.match(source)
            if m:
                yield source, int(m.group('modno'))

    @classmethod
    def _data_is_raw(cls, data, source: str):
        # For most detectors, raw data is uint16 & corrected is float32.
        # Overridden for AGIPD, where output dtype is configurable.
        kd = data[source, cls._main_data_key]
        return np.issubdtype(kd.dtype, np.integer)

    @classmethod
    def _identify_sources(cls, data, detector_name, modules=None, raw=None):
        if raw is True:
            pat = re.escape(detector_name) + cls._source_raw_pat
            source_to_modno = dict(cls._source_matches(data, pat))
            if not all(cls._data_is_raw(data, s) for s in source_to_modno):
                # Older corrected data used the same names as raw
                raise ValueError(
                    f"Raw data was not found: {detector_name}/DET/... sources "
                    f"are from corrected data"
                )

        else:
            # Prefer corrected data
            pat = re.escape(detector_name) + cls._source_corr_pat
            source_to_modno = dict(cls._source_matches(data, pat))

            if not source_to_modno:
                # Data named like raw may also be proc
                pat = re.escape(detector_name) + cls._source_raw_pat
                source_to_modno = dict(cls._source_matches(data, pat))

                if (raw is False) and any(cls._data_is_raw(data, s) for s in source_to_modno):
                    raise SourceNameError(f'{detector_name}/CORR/...')
                # raw=None -> legacy behaviour: prefer corrected but allow raw

        if modules is not None:
            source_to_modno = {s: n for (s, n) in source_to_modno.items()
                               if n in modules}

        if not source_to_modno:
            dc = '(DET|CORR)' if raw is None else 'DET' if raw else 'CORR'
            raise SourceNameError(f'{detector_name}/{dc}/...')

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
    def _split_align_chunk(chunk, target_train_ids: np.ndarray, length_limit=np.inf):
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
        length_limit: int
          Maximum length of slices (stop - start) to yield. Larger slices will
          be split up into several pieces. Unlimited by default.
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

            # Split the matched data if needed for length_limit
            n_batches = max(math.ceil(n_match / length_limit), 1)
            for i in range(n_batches):
                start = i      * n_match // n_batches
                stop = (i + 1) * n_match // n_batches
                yield (slice(tgt_start + start, tgt_start + stop),
                       slice(chunk_match_start + start, chunk_match_start + stop))

            # Prepare remaining data in the chunk for the next match
            chunk_match_start += n_match
            chunk_tids = chunk_tids[n_match:]

    @property
    def train_ids(self):
        return self.data.train_ids

    @property
    def train_id_chunks(self):
        # Used to be used internally. Kept temporarily in case anyone else used it.
        warn(
            "detector.train_id_chunks is likely to be removed in the future. "
            "Please contact da-support@xfel.eu if you're using it",
            stacklevel=2
        )
        train_id_arr = np.asarray(self.data.train_ids)
        split_indices = np.where(np.diff(train_id_arr) != 1)[0] + 1
        return np.split(train_id_arr, split_indices)

    @property
    def train_id_to_ix(self):
        # Used to be used internally. Kept temporarily in case anyone else used it.
        warn(
            "detector.train_id_to_ix is likely to be removed in the future. "
            "Please contact da-support@xfel.eu if you're using it",
            stacklevel=2
        )
        # Cumulative sum gives the end of each train, subtract to get start
        return self.frame_counts.cumsum() - self.frame_counts

    @property
    def frames_per_train(self):
        counts = set(self.frame_counts.unique()) - {0}
        if len(counts) > 1:
            raise ValueError(f"Varying number of frames per train: {counts}")
        return counts.pop() * self._frames_per_entry

    def __repr__(self):
        # Show raw/proc
        det = type(self).__name__
        raw = all(self._data_is_raw(self.data, s) for s in self.source_to_modno)
        rp = 'raw' if raw else 'proc'
        return (f"<{det}: Data interface for detector {self.detector_name!r} "
                f"- {rp} data with {len(self.source_to_modno)} modules>")

    def select_trains(self, trains):
        """Select a subset of trains from this data as a new object.

        Slice trains by position within this data::

            sel = det.select_trains(np.s_[:5])

        Or select trains by train ID, with a slice or a list::

            from extra_data import by_id
            sel1 = det.select_trains(by_id[142844490 : 142844495])
            sel2 = det.select_trains(by_id[[142844490, 142844493, 142844494]])
        """
        # Using a copy to bypass the source & train checks in __init__
        res = copy(self)
        res.data = self.data.select_trains(trains)
        res.frame_counts = self.frame_counts[res.data.train_ids]
        res.train_ids_perframe = np.repeat(
            res.frame_counts.index.values, res.frame_counts.values.astype(np.intp)
        )
        return res

    def split_trains(self, parts=None, trains_per_part=None, frames_per_part=None):
        """Split this data into chunks with a fraction of the trains each.

        At least one of *parts*, *trains_per_part* or *frames_per_part* must be
        specified. You can pass any combination of these.

        Parameters
        ----------

        parts: int
            How many parts to split the data into. If trains_per_part is also
            specified, this is a minimum, and it may make more parts.
            It may also make fewer if there are fewer trains in the data.
        trains_per_part: int
            A maximum number of trains in each part. Parts will often have
            fewer trains than this.
        frames_per_part: int
            A target number of frames in each part. Each chunk should have up
            to this many frames, but chunks always contain complete trains,
            so if this is less than one train, you may get single train chunks
            with more frames. When ``frames_per_part`` is used, the final
            chunk may be much smaller than the others.
        """
        if {parts, trains_per_part, frames_per_part} == {None}:
            raise ValueError(
                "One of parts, trains_per_part, frames_per_part must be specified"
            )
        if frames_per_part is None:
            for s in split_trains(len(self.train_ids), parts, trains_per_part):
                yield self.select_trains(s)
        else:
            # frames_per_part was specified. We don't assume that the number
            # of frames per train is constant, so we'll iterate over trains
            # and cut off each chunk when we reach the relevant number.
            if not self.train_ids:
                return  # No data to split

            if trains_per_part is None:
                trains_per_part = np.inf
            if parts:
                trains_per_part = min(trains_per_part, len(self.train_ids) // parts)

            chunk_start = 0
            ntrains = 1
            nentries = self.frame_counts.iloc[0]

            for frame_ct in self.frame_counts.iloc[1:]:
                ntrains += 1
                nentries += frame_ct
                if (ntrains > trains_per_part) or (nentries * self._frames_per_entry > frames_per_part):
                    # We've got a full chunk
                    chunk_end = chunk_start + ntrains - 1
                    yield self.select_trains(np.s_[chunk_start:chunk_end])
                    chunk_start = chunk_end
                    ntrains = 1
                    nentries = frame_ct

            # There will always be at least the last train left to yield
            yield self.select_trains(np.s_[chunk_start:])

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
        return self[key].xarray(fill_value=fill_value, roi=roi, astype=astype)


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
        return self[key].dask_array(labelled=True, fill_value=fill_value, astype=astype)

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

    def data_availability(self, module_gaps=False):
        """Get an array indicating what image data is available

        Returns a boolean array (modules, entries), True where a module has data
        for a given train, False for missing data.
        """
        return self[self._main_data_key].data_availability(module_gaps)


class XtdfDetectorBase(MultimodDetectorBase):
    """Common machinery for a group of detectors with similar data format

    AGIPD, DSSC & LPD all store pulse-resolved data in an "image" group,
    with both trains and pulses along the first dimension. This allows a
    different number of frames to be stored for each train, which makes
    access more complicated.
    """
    n_modules = 16
    _main_data_key = 'image.data'
    _mask_data_key = 'image.mask'

    def __getitem__(self, item):
        if item.startswith('image.'):
            return XtdfImageMultimodKeyData(self, item)
        return super().__getitem__(item)

    def masked_data(self, key=None, *, mask_bits=None, masked_value=np.nan):
        """Combine corrected data with the mask in the files

        This provides an interface similar to ``det['image.data']``, but masking
        out pixels with the mask from the correction pipeline.

        Parameters
        ----------

        key: str
          The data key to look at, by default the main data key of the detector
          (e.g. 'image.data').
        mask_bits: int or list of ints
          Reasons to exclude pixels, as a bitmask or a list of integers.
          By default, all types of bad pixel are masked out.
        masked_value: int, float
          The replacement value to use for masked data. By default this is NaN.
        """
        key = key or self._main_data_key
        assert key.startswith('image.')
        if self._mask_data_key not in self:
            raise RuntimeError(
                f"This data doesn't include a mask ({self._mask_data_key}). "
                f"You might be using raw instead of corrected data."
            )
        if isinstance(mask_bits, Iterable):
            mask_bits = self._combine_bitfield(mask_bits)
        return XtdfMaskedKeyData(
            self, key, mask_key=self._mask_data_key,
            mask_bits=mask_bits, masked_value=masked_value
        )

    # Several methods below are overridden in LPD1M for parallel gain mode

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
            return self[key].select_pulses(pulses).xarray(
                fill_value=fill_value, roi=roi, subtrain_index=subtrain_index,
                astype=astype, unstack_pulses=unstack_pulses,
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
        from xarray import DataArray
        if subtrain_index not in {'pulseId', 'cellId'}:
            raise ValueError("subtrain_index must be 'pulseId' or 'cellId'")
        if key.startswith('image.'):
            arr = self[key].dask_array(
                labelled=True, subtrain_index=subtrain_index,
                fill_value=fill_value, astype=astype
            )
            # Preserve the quirks of this method before refactoring
            if self[key]._extraneous_dim:
                arr = arr.expand_dims('tmp_name', axis=2)
            frame_idx = arr.indexes['train_pulse'].set_names(
                ['trainId', subtrain_index], level=[0, -1]
            )
            dims = ['module', 'train_pulse'] + [f'dim_{i}' for i in range(arr.ndim - 2)]
            return DataArray(arr.data, dims=dims, coords={
                'train_pulse': frame_idx, 'module': arr.indexes['module'],
            })
        else:
            return super().get_dask_array(key, fill_value=fill_value, astype=astype)

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


class MultimodKeyData:
    def __init__(self, det: MultimodDetectorBase, key):
        self.det = det
        self.key = key
        self.modno_to_keydata = {
            m: det.data[s, key] for (m, s) in det.modno_to_source.items()
        }

    def _init_kwargs(self):  # Extended in subclasses
        return dict(det=self.det, key=self.key)

    @property
    def train_ids(self):
        return self.det.train_ids

    def train_id_coordinates(self):
        return np.array(self.det.train_ids)

    @property
    def modules(self):
        return sorted(self.modno_to_keydata)

    @property
    def _eg_keydata(self):
        return self.modno_to_keydata[min(self.modno_to_keydata)]

    @property
    def ndim(self):
        return self._eg_keydata.ndim + 1

    def buffer_shape(self, module_gaps=False, roi=()):
        """Get the array shape for this data

        If *module_gaps* is True, include space for modules which are missing
        from the data. *roi* may be a tuple of slices defining a region of
        interest on the inner dimensions of the data.
        """
        module_dim = self.det.n_modules if module_gaps else len(self.modno_to_keydata)

        return ((module_dim, len(self.train_ids))
                # Shape of 1 frame for 1 module with the ROI applied:
                + roi_shape(self._eg_keydata.entry_shape, roi))

    @property
    def shape(self):
        return self.buffer_shape()

    @property
    def dimensions(self):
        return ['module', 'trainId'] + ['dim_%d' % i for i in range(self.ndim - 2)]

    @property
    def dtype(self):
        return self._eg_keydata.dtype

    # For select_trains() & split_trains() to work correctly with subclasses
    def _with_selected_det(self, det_selected):
        kw = self._init_kwargs()
        kw.update(det=det_selected)
        return type(self)(**kw)

    def select_trains(self, trains):
        return self._with_selected_det(self.det.select_trains(trains))

    def __getitem__(self, item):
        return self.select_trains(item)

    __iter__ = None  # Disable iteration

    def split_trains(self, parts=None, trains_per_part=None, frames_per_part=None):
        for det_split in self.det.split_trains(parts, trains_per_part, frames_per_part):
            yield self._with_selected_det(det_split)

    def ndarray(self, *, fill_value=None, out=None, roi=(), astype=None, module_gaps=False):
        """Get data as a plain NumPy array with no labels"""
        train_ids = np.asarray(self.det.train_ids)

        out_shape = self.buffer_shape(module_gaps, roi)

        if out is None:
            dtype = self._eg_keydata.dtype if astype is None else np.dtype(astype)
            out = _out_array(out_shape, dtype, fill_value=fill_value)
        elif out.shape != out_shape:
            raise ValueError(f'requires output array of shape {out_shape}')

        for i, (modno, kd) in enumerate(sorted(self.modno_to_keydata.items())):
            mod_ix = (modno - self.det._modnos_start_at) if module_gaps else i
            for chunk in kd._data_chunks:
                for tgt_slice, chunk_slice in self.det._split_align_chunk(chunk, train_ids):
                    chunk.dataset.read_direct(
                        out[mod_ix, tgt_slice], source_sel=(chunk_slice,) + roi
                    )
        return out

    def _wrap_xarray(self, arr):
        from xarray import DataArray

        coords = {'module': self.modules, 'trainId': self.train_id_coordinates()}
        return DataArray(arr, dims=self.dimensions, coords=coords)

    def xarray(self, *, fill_value=None, roi=(), astype=None):
        arr = self.ndarray(fill_value=fill_value, roi=roi, astype=astype)
        return self._wrap_xarray(arr)

    def dask_array(self, *, labelled=False, fill_value=None, astype=None):
        from dask.delayed import delayed
        from dask.array import concatenate, from_delayed

        entry_size = (self.dtype.itemsize *
            len(self.modno_to_keydata) * np.prod(self._eg_keydata.entry_shape)
        )
        # Aim for 1GB chunks, with an arbitrary maximum of 256 trains
        split = self.split_trains(frames_per_part=min(1024 ** 3 / entry_size, 256))

        arr = concatenate([from_delayed(
            delayed(c.ndarray)(fill_value=fill_value, astype=astype),
            shape=c.shape, dtype=self.dtype
        ) for c in split], axis=1)

        if labelled:
            return self._wrap_xarray(arr)

        return arr

    def data_availability(self, module_gaps=False):
        """Get an array indicating what data is available

        Returns a boolean array (modules, entries), True where a module has data
        for a given train, False for missing data.
        """
        train_ids = self.train_id_coordinates()

        module_dim = self.det.n_modules if module_gaps else len(self.modno_to_keydata)
        out = np.zeros((module_dim, len(train_ids)), dtype=np.bool_)

        for i, (modno, kd) in enumerate(sorted(self.modno_to_keydata.items())):
            mod_ix = (modno - self.det._modnos_start_at) if module_gaps else i
            for chunk in kd._data_chunks:
                for tgt_slice, _ in self.det._split_align_chunk(chunk, train_ids):
                    out[mod_ix, tgt_slice] = True

        return out


class DetectorMaskedKeyData(MultimodKeyData):
    def __init__(self, *args, mask_key, mask_bits, masked_value, **kwargs):
        super().__init__(*args, **kwargs)
        self._mask_key = mask_key
        self._mask_bits = mask_bits
        self._masked_value = masked_value

    def __repr__(self):
        return f"<Masked {self.key!r} detector data for {len(self.modules)} modules>"

    def _init_kwargs(self):
        kw = super()._init_kwargs()
        kw.update(
            mask_key=self._mask_key,
            mask_bits=self._mask_bits,
            masked_value=self._masked_value,
        )
        return kw

    # Overridden for XTDF data to accommodate pulse selection
    def _mask_keydata(self):
        return self.det[self._mask_key]

    def _load_mask(self, module_gaps):
        """Load the mask & convert to boolean (True for bad pixels)"""
        mask_data = self._mask_keydata().ndarray(module_gaps=module_gaps)
        if self._mask_bits is None:
            return mask_data != 0  # Skip extra temporary array from &
        else:
            return (mask_data & self._mask_bits) != 0

    def ndarray(self, *, module_gaps=False, **kwargs):
        """Load data into a NumPy array & apply the mask"""
        # Load mask first: it shrinks from 4 bytes/px to 1, so peak memory use
        # is lower than loading it after the data
        mask = self._load_mask(module_gaps=module_gaps)

        data = super().ndarray(module_gaps=module_gaps, **kwargs)
        data[mask] = self._masked_value
        return data


class XtdfImageMultimodKeyData(MultimodKeyData):
    _sel_frames_cached = None
    det: XtdfDetectorBase

    def __init__(self, det: XtdfDetectorBase, key, pulse_sel=by_index[0:MAX_PULSES:1]):
        super().__init__(det, key)
        self._pulse_sel = pulse_sel
        entry_shape = self._eg_keydata.entry_shape
        self._extraneous_dim = (len(entry_shape) >= 1) and (entry_shape[0] == 1)

    def _init_kwargs(self):
        kw = super()._init_kwargs()
        kw.update(pulse_sel=self._pulse_sel)
        return kw

    @property
    def ndim(self):
        return super().ndim - (1 if self._extraneous_dim else 0)

    def _all_pulses(self):
        psv = self._pulse_sel.value
        return isinstance(psv, slice) and psv == slice(0, MAX_PULSES, 1)

    def buffer_shape(self, module_gaps=False, roi=()):
        """Get the array shape for this data

        If *module_gaps* is True, include space for modules which are missing
        from the data. *roi* may be a tuple of slices defining a region of
        interest on the inner dimensions of the data.
        """
        module_dim = self.det.n_modules if module_gaps else len(self.modno_to_keydata)

        # len(self.train_id_coordinates()), but avoids allocating extra arrays
        if self._all_pulses():
            nframes_sel = len(self.det.train_ids_perframe)
        else:
            nframes_sel = int(self._sel_frames.sum())

        entry_shape = self._eg_keydata.entry_shape
        if self._extraneous_dim:
            entry_shape = entry_shape[1:]

        return (module_dim, nframes_sel) + roi_shape(entry_shape, roi)

    @property
    def shape(self):
        return self.buffer_shape()

    def train_id_coordinates(self):
        # XTDF 'image' group can have >1 entry per train
        a = self.det.train_ids_perframe
        # Only allocate sel_frames array if we need it:
        if not self._all_pulses():
            a = a[self._sel_frames]
        else:
            a = a.copy()  # So you can't accidentally modify the internal array
        return a

    def pulse_id_coordinates(self):
        """Get an array of pulse IDs per-frame for this data"""
        a = self.det._collect_inner_ids('pulseId')
        if not self._all_pulses():
            a = a[self._sel_frames]
        return a

    def cell_id_coordinates(self):
        """Get an array of memory cell IDs per-frame for this data"""
        a = self.det._collect_inner_ids('cellId')
        if not self._all_pulses():
            a = a[self._sel_frames]
        return a

    @property
    def dimensions(self):
        ndim_inner = self.ndim - 2
        # TODO: this assumes we can tell what the axes are just from the
        # number of dimensions. Works for the data we've seen, but we
        # should look for a more reliable way.
        if ndim_inner == 3:
            # image.data in raw data
            entry_dims = ['data_gain', 'slow_scan', 'fast_scan']
        elif ndim_inner == 2:
            # image.data, image.gain, image.mask in calibrated data
            entry_dims = ['slow_scan', 'fast_scan']
        else:
            # Everything else seems to be 1D, but just in case
            entry_dims = [f'dim_{i}' for i in range(ndim_inner)]
        return ['module', 'train_pulse'] + entry_dims

    def select_pulses(self, pulses):
        kw = self._init_kwargs()
        kw.update(pulse_sel=_check_pulse_selection(pulses))
        return type(self)(**kw)

    @property
    def _sel_frames(self):
        if self._sel_frames_cached is None:
            p = self._pulse_sel
            if isinstance(p, by_index):
                if self._all_pulses():
                    s = np.ones(len(self.det.train_ids_perframe), np.bool_)
                else:
                    s = self.det._select_pulse_indices(p, self.det.frame_counts)
            elif isinstance(p, by_id):
                pulse_ids = self.det._collect_inner_ids('pulseId')
                s = _select_pulse_ids(p, pulse_ids)
            else:
                raise TypeError(f"Pulse selection should not be {type(p)}")
            self._sel_frames_cached = s
        return self._sel_frames_cached

    def _read_chunk(self, chunk: DataChunk, mod_out, roi):
        """Read per-pulse data from file into an output array (of 1 module)"""
        # Limit to 5 GB sections of the dataset at once, so the temporary
        # arrays used in the workaround below are not too large.
        nbytes_frame = chunk.dataset.dtype.itemsize
        for dim in chunk.dataset.shape[1:]:
            nbytes_frame *= dim
        frame_limit = 5 * (1024 ** 3) // nbytes_frame

        for tgt_slice, chunk_slice in self.det._split_align_chunk(
                chunk, self.det.train_ids_perframe, length_limit=frame_limit
        ):
            inc_pulses_chunk = self._sel_frames[tgt_slice]
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
            # shape as the full chunk, reads into it, and then copies the selected
            # data to the output array. The extra memory copy is not optimal,
            # but it's better than the HDF5 performance issue, at least in some
            # realistic cases.
            # N.B. tmp should only use memory for the data it contains -
            # zeros() uses calloc, so the OS can do virtual memory tricks.
            # Don't change this to zeros_like() !
            tmp = np.zeros(
                shape=inc_pulses_chunk.shape + chunk.dataset.shape[1:],
                dtype=chunk.dataset.dtype
            )
            tmp_sel = np.nonzero(inc_pulses_chunk)[0]
            dataset_sel = tmp_sel + chunk_slice.start
            chunk.dataset.read_direct(
                tmp, source_sel=(dataset_sel,) + roi, dest_sel=(tmp_sel,) + roi,
            )
            # Where does this data go in the target array?
            tgt_start_ix = self._sel_frames[:tgt_slice.start].sum()
            tgt_pulse_sel = slice(
                tgt_start_ix, tgt_start_ix + inc_pulses_chunk.sum()
            )
            # Copy data from temp array to output array
            np.compress(
                inc_pulses_chunk, tmp[np.index_exp[:] + roi],
                axis=0, out=mod_out[tgt_pulse_sel]
            )

    def _read_parallel_decompress(self, out, module_gaps, threads=16):
        try:
            from .compression import multi_dataset_decompressor, parallel_decompress_chunks
        except ImportError:
            return False

        modno_to_keydata_no_virtual = {}
        all_datasets = []
        for (m, vkd) in self.modno_to_keydata.items():
            modno_to_keydata_no_virtual[m] = kd = vkd._without_virtual_overview()
            all_datasets.extend([f.file[kd.hdf5_data_path] for f in kd.files])

        if any(d.chunks != (1,) + d.shape[1:] for d in all_datasets):
            return False  # Chunking not as we expect

        decomp_proto = multi_dataset_decompressor(all_datasets)
        if decomp_proto is None:
            return False  # No suitable fast decompression path

        load_tasks = []
        for i, (modno, kd) in enumerate(sorted(modno_to_keydata_no_virtual.items())):
            mod_ix = (modno - self.det._modnos_start_at) if module_gaps else i
            # 'chunk' in the lines below means a range of consecutive indices
            # in one HDF5 dataset, as elsewhere in EXtra-data.
            # We use this to build a list of HDF5 chunks (1 frame per chunk)
            # to be loaded & decompressed. Sorry about that.
            for chunk in kd._data_chunks:
                dset = chunk.dataset

                for tgt_slice, chunk_slice in self.det._split_align_chunk(
                        chunk, self.det.train_ids_perframe,
                ):
                    inc_pulses_chunk = self._sel_frames[tgt_slice]
                    if inc_pulses_chunk.sum() == 0:  # No data from this chunk selected
                        continue

                    dataset_ixs = np.nonzero(inc_pulses_chunk)[0] + chunk_slice.start

                    # Where does this data go in the target array?
                    tgt_start_ix = self._sel_frames[:tgt_slice.start].sum()

                    # Each task is a h5py.h5d.DatasetID, coordinates & array destination
                    load_tasks.extend([
                        (dset.id, (ds_ix, 0, 0), out[mod_ix, tgt_start_ix + i])
                        for i, ds_ix in enumerate(dataset_ixs)]
                    )

        parallel_decompress_chunks(load_tasks, decomp_proto, threads=threads)

        return True

    def ndarray(self, *, fill_value=None, out=None, roi=(), astype=None,
                module_gaps=False, decompress_threads=None):
        """Get an array of per-pulse data (image.*) for xtdf detector"""
        out_shape = self.buffer_shape(module_gaps=module_gaps, roi=roi)

        if out is None:
            dtype = self._eg_keydata.dtype if astype is None else np.dtype(astype)
            out = _out_array(out_shape, dtype, fill_value=fill_value)
        elif out.shape != out_shape:
            raise ValueError(f'requires output array of shape {out_shape}')

        if roi == () and astype is None:
            if decompress_threads is None:
                decompress_threads = default_num_threads(fixed_limit=16)

            if decompress_threads > 1:
                if self._read_parallel_decompress(out, module_gaps, decompress_threads):
                    return out

        reading_view = out.view()
        if self._extraneous_dim:
            reading_view.shape = out.shape[:2] + (1,) + out.shape[2:]
            # Ensure ROI applies to pixel dimensions, not the extra
            # dim in raw data (except AGIPD, where it is data/gain)
            roi = np.index_exp[:] + roi

        for i, (modno, kd) in enumerate(sorted(self.modno_to_keydata.items())):
            mod_ix = (modno - self.det._modnos_start_at) if module_gaps else i
            for chunk in kd._data_chunks:
                self._read_chunk(chunk, reading_view[mod_ix], roi)

        return out

    def _wrap_xarray(self, arr, subtrain_index='pulseId'):
        from xarray import DataArray
        inner_ids = self.det._collect_inner_ids(subtrain_index)
        index = self.det._make_image_index(
            self.det.train_ids_perframe, inner_ids, subtrain_index[:-2]
        )[self._sel_frames]

        return DataArray(arr, dims=self.dimensions, coords={
            'train_pulse': index, 'module': self.modules,
        })

    def xarray(self, *, pulses=None, fill_value=None, roi=(), astype=None,
               subtrain_index='pulseId', unstack_pulses=False, decompress_threads=None):
        arr = self.ndarray(
            fill_value=fill_value,
            roi=roi,
            astype=astype,
            decompress_threads=decompress_threads,
        )
        out = self._wrap_xarray(arr, subtrain_index)

        if unstack_pulses:
            # Separate train & pulse dimensions, and arrange dimensions
            # so that the data is contiguous in memory.
            dim_order = ['module'] + out.indexes['train_pulse'].names + self.dimensions[2:]
            return out.unstack('train_pulse').transpose(*dim_order)

        return out

    def dask_array(self, *, labelled=False, subtrain_index='pulseId',
                   fill_value=None, astype=None, frames_per_chunk=None):
        from dask.delayed import delayed
        from dask.array import concatenate, from_delayed

        entry_size = (self.dtype.itemsize *
            len(self.modno_to_keydata) * np.prod(self._eg_keydata.entry_shape)
        )
        if frames_per_chunk is None:
            # Aim for 2GB chunks, with an arbitrary maximum of 1024 frames
            frames_per_chunk = min(2 * 1024 ** 3 / entry_size, 1024)
        split = self.split_trains(frames_per_part=frames_per_chunk)

        arr = concatenate([from_delayed(
            delayed(c.ndarray)(fill_value=fill_value, astype=astype),
            shape=c.shape, dtype=self.dtype
        ) for c in split], axis=1)

        if labelled:
            return self._wrap_xarray(arr, subtrain_index)

        return arr


class XtdfMaskedKeyData(DetectorMaskedKeyData, XtdfImageMultimodKeyData):
    # Created from xtdf_det.masked_data()
    def _mask_keydata(self):
        return self.det[self._mask_key].select_pulses(self._pulse_sel)


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
            self.data[source].files,
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
        from xarray import DataArray
        file, pos, ds = self._find_data(source, key, tid)
        if file is None:
            return None

        group = key.partition('.')[0]
        firsts, counts = file.get_index(source, group)
        first, count = firsts[pos], counts[pos]
        if count == 1:
            return DataArray(ds[first])
        else:
            return DataArray(ds[first : first + count])

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
        from xarray import DataArray
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
        elif isinstance(self.pulses, by_index):
            positions = self._select_pulse_indices(count)
        else:
            raise TypeError(f"Pulse selection should not be {type(self.pulses)}")
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

        data = ds[data_positions]

        # Raw files have a spurious extra dimension
        if data.ndim >= 2 and data.shape[1] == 1:
            data = data[:, 0]

        dims = self.data[key].dimensions[1:]  # excluding 'module' dim
        coords = {'train_pulse': train_pulse_ids}

        arr = DataArray(data, coords=coords, dims=dims)

        # Separate train & pulse dimensions, and arrange dimensions
        # so that the data is contiguous in memory.
        dim_order = train_pulse_ids.names + dims[1:]
        return arr.unstack('train_pulse').transpose(*dim_order)

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

        return np.nonzero(np.isin(pulse_ids, desired))[0].astype(np.uint64)

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
        import xarray
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
    raw: bool
      True to access raw data, False for corrected. The default is to use
      corrected if available & raw otherwise.
    """
    _det_name_pat = r'[^/]+_AGIPD1M[^/]*'
    _source_raw_pat = r'/DET/(?P<modno>\d+)CH'
    _source_corr_pat = r'/CORR/(?P<modno>\d+)CH'
    module_shape = (512, 128)

    @classmethod
    def _data_is_raw(cls, data, source: str):
        # Raw AGIPD data has an extra dimension (data/gain) compared to raw
        kd = data[source, cls._main_data_key]
        return kd.ndim == 4


@multimod_detectors
class AGIPD500K(XtdfDetectorBase):
    """An interface to AGIPD-500K data

    Detector names are like 'HED_DET_AGIPD500K2G', otherwise this is identical
    to :class:`AGIPD1M`.
    """
    _det_name_pat = r'[^/]+AGIPD500K[^/]*'
    _source_raw_pat = r'/DET/(?P<modno>\d+)CH'
    _source_corr_pat = r'/CORR/(?P<modno>\d+)CH'
    module_shape = (512, 128)
    n_modules = 8

    @classmethod
    def _data_is_raw(cls, data, source: str):
        # Raw AGIPD data has an extra dimension (data/gain) compared to raw
        kd = data[source, cls._main_data_key]
        return kd.ndim == 4


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
    raw: bool
      True to access raw data, False for corrected. The default is to use
      corrected if available & raw otherwise.
    """
    _det_name_pat = r'[^/]+_DSSC1M[^/]*'
    _source_raw_pat = r'/DET/(?P<modno>\d+)CH'
    _source_corr_pat = r'/CORR/(?P<modno>\d+)CH'
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
    raw: bool
      True to access raw data, False for corrected. The default is to use
      corrected if available & raw otherwise.
    parallel_gain: bool
      Set to True to read this data as parallel gain data, where high, medium
      and low gain data are stored sequentially within each train. This will
      repeat the pulse & cell IDs from the first 1/3 of each train, and add gain
      stage labels from 0 (high-gain) to 2 (low-gain).
    """
    _det_name_pat = r'[^/]+_LPD1M[^/]*'
    _source_raw_pat = r'/DET/(?P<modno>\d+)CH'
    _source_corr_pat = r'/CORR/(?P<modno>\d+)CH'
    module_shape = (256, 256)

    def __init__(self, data: DataCollection, detector_name=None, modules=None,
                 *, min_modules=1, parallel_gain=False, raw=None):
        super().__init__(
            data, detector_name, modules, min_modules=min_modules, raw=raw
        )

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
    first_modno: int
      The module number in the source name for the first detector module.
      e.g. FXE_XAD_JF500K/DET/JNGFR03:daqOutput should have first_modno = 3
    raw: bool
      True to access raw data, False for corrected. The default is to use
      corrected if available & raw otherwise.
    """
    # We appear to have a few different formats for source names:
    # SPB_IRDA_JNGFR/DET/MODULE_1:daqOutput  (e.g. in p 2566, r 61)
    # SPB_IRDA_JF4M/DET/JNGFR03:daqOutput    (e.g. in p 2732, r 12)
    # FXE_XAD_JF500K/DET/JNGFR03:daqOutput    (e.g. in p 2478, r 52)
    # HED_IA1_JF500K1/DET/JNGFR01:daqOutput    (e.g. in p 2656, r 230)
    # FXE_XAD_JF1M/DET/RECEIVER-1
    _det_name_pat = r'[^/]+_(JNGFR|JF[14]M|JUNGF|JF|JF500K\d?)'
    _source_raw_pat = r'/DET/(MODULE_|RECEIVER-|JNGFR)(?P<modno>\d+)'
    _source_corr_pat = r'/CORR/(MODULE_|RECEIVER-|JNGFR)(?P<modno>\d+)'
    _main_data_key = 'data.adc'
    _mask_data_key = 'data.mask'
    _modnos_start_at = 1
    module_shape = (512, 1024)

    def __init__(self, data: DataCollection, detector_name=None, modules=None,
                 *, min_modules=1, n_modules=None, first_modno=1, raw=None):
        super().__init__(data, detector_name, modules, min_modules=min_modules, raw=raw)

        # Overwrite modno based on given starting module number and update
        # source_to_modno and modno_to_source.
        # JUNGFRAU modno is expected (e.g. extra_geom) to start with 1.
        self.source_to_modno = {s: (m - first_modno + 1)
                                for (s, m) in self.source_to_modno.items()}
        self.modno_to_source = {m: s for (s, m) in self.source_to_modno.items()}

        if n_modules is not None:
            self.n_modules = int(n_modules)
        else:
            # Re-scan sources without modules= selection to find largest number
            self.n_modules = max(
                self._identify_sources(data, self.detector_name, raw=raw).values()
            ) - first_modno + 1

        # In burst mode, JUNGFRAU can have 16 frames per train
        src = next(iter(self.source_to_modno))
        self._frames_per_entry = self.data[src, self._main_data_key].entry_shape[0]

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

    def cell_ids(self):
        MISSING = 255  # To fit in uint8
        cids = self.select_trains(np.s_[:1])['data.memoryCell'].ndarray(
            fill_value=MISSING
        )[:, 0] # -> (modules, cells)

        cells_min = cids.min(axis=0)
        if (cells_min == MISSING).any():
            raise Exception(f"Failed to find memoryCell")
        cids[cids == MISSING] = 0
        if (cells_min != cids.max(axis=0)).any():
            raise Exception(f"Inconsistent memoryCell for different modules")

        # Pulse IDs make sense. Drop the modules dimension, giving one
        # pulse ID for each frame.
        return cells_min


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
        for name in cls._find_detector_names(data):
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
