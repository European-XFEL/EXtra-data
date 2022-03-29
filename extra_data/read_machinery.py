"""Machinery for reading Karabo HDF5 files

The public API is in extra_data.reader; this is internal code.
"""
import logging
import math
import os.path as osp
import re
import time
from glob import iglob
from warnings import warn

import numpy as np

log = logging.getLogger(__name__)

DETECTOR_NAMES = {'AGIPD', 'DSSC', 'LPD'}
DETECTOR_SOURCE_RE = re.compile(r'(.+)/DET/(\d+)CH')

DATA_ROOT_DIR = '/gpfs/exfel/exp'


class _SliceConstructor(type):
    """Allows instantiation like subclass[1:5]
    """

    def __getitem__(self, item):
        return self(item)


class _SliceConstructable(metaclass=_SliceConstructor):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        indices = self.value
        if not isinstance(indices, tuple):
            indices = (indices,)

        return "{}[{}]".format(
            type(self).__name__, ', '.join(self._indexing_repr(v) for v in indices)
        )

    @staticmethod
    def _indexing_repr(value):
        """Represent values as used in canonical slicing syntax"""
        if value is Ellipsis:
            return '...'
        elif isinstance(value, slice):
            start = value.start if (value.start is not None) else ''
            stop = value.stop if (value.stop is not None) else ''
            step = ':{}'.format(value.step) if (value.step is not None) else ''
            return '{}:{}{}'.format(start, stop, step)

        return repr(value)


class by_id(_SliceConstructable):
    pass


class by_index(_SliceConstructable):
    pass


def _tid_to_slice_ix(tid, train_ids, stop=False):
    """Convert a train ID to an integer index for slicing the dataset

    Throws ValueError if the slice won't overlap the trains in the data.
    The *stop* parameter tells it which end of the slice it is making.
    """
    if tid is None:
        return None

    try:
        return train_ids.index(tid)
    except ValueError:
        pass

    if len(train_ids) == 0:
        warn("Using train ID slice on data with no trains selected", stacklevel=4)
        return 0

    if tid < train_ids[0]:
        if stop:
            warn(
                f"Train ID {tid} is before this run (starts at {train_ids[0]})",
                stacklevel=4,
            )
            return 0
        else:
            return None
    elif tid > train_ids[-1]:
        if stop:
            return None
        else:
            warn(
                f"Train ID {tid} is after this run (ends at {train_ids[-1]})",
                stacklevel=4,
            )
            return len(train_ids)
    else:
        # This train ID is within the run, but doesn't have an entry.
        # Find the first ID in the run greater than the one given.
        return (train_ids > tid).nonzero()[0][0]


def select_train_ids(train_ids, sel):
    if isinstance(sel, by_index):
        sel = sel.value
    elif isinstance(sel, int):
        sel = slice(sel, sel+1, None)

    if isinstance(sel, by_id) and isinstance(sel.value, slice):
        # Slice by train IDs
        start_ix = _tid_to_slice_ix(sel.value.start, train_ids, stop=False)
        stop_ix = _tid_to_slice_ix(sel.value.stop, train_ids, stop=True)
        return train_ids[start_ix: stop_ix: sel.value.step]
    elif isinstance(sel, by_id) and isinstance(sel.value, (list, np.ndarray)):
        # Select a list of trains by train ID
        new_train_ids = sorted(set(train_ids).intersection(sel.value))
        if len(sel.value) and not new_train_ids:
            warn(
                f"Given train IDs not found among {len(train_ids)} trains in "
                "collection", stacklevel=3,
            )
        return new_train_ids
    elif isinstance(sel, slice):
        # Slice by indexes in this collection
        return train_ids[sel]
    elif isinstance(sel, (list, np.ndarray)):
        # Select a list of trains by index in this collection
        return sorted(np.asarray(train_ids)[sel])
    else:
        raise TypeError(type(sel))


def split_trains(n_trains, parts=None, trains_per_part=None) -> [slice]:
    if trains_per_part is not None:
        assert trains_per_part >= 1
        n_parts = math.ceil(n_trains / trains_per_part)
        if parts is not None:
            n_parts = max(n_parts, min(parts, n_trains))
    elif parts is not None:
        assert parts >= 1
        n_parts = min(parts, n_trains)
    else:
        raise ValueError("Either parts or trains_per_part must be specified")

    return [
        slice(i * n_trains // n_parts, (i + 1) * n_trains // n_parts)
        for i in range(n_parts)
    ]

class DataChunk:
    """Reference to a contiguous chunk of data for one or more trains."""
    def __init__(self, file, dataset_path, first, train_ids, counts):
        self.file = file
        self.dataset_path = dataset_path
        self.first = first
        self.train_ids = train_ids
        self.counts = counts

    @property
    def slice(self):
        return slice(self.first, self.first + np.sum(self.counts))

    @property
    def total_count(self):
        return int(np.sum(self.counts, dtype=np.uint64))

    @property
    def dataset(self):
        return self.file.file[self.dataset_path]


# contiguous_regions() by Joe Kington on Stackoverflow
# https://stackoverflow.com/a/4495197/434217
# Used here under Stackoverflow's default CC-BY-SA 3.0 license.
def contiguous_regions(condition):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index."""

    # Find the indices of changes in "condition"
    d = np.diff(condition)
    idx, = d.nonzero()

    # We need to start things after the change in "condition". Therefore,
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size] # Edit

    # Reshape the result into two columns
    idx.shape = (-1,2)
    return idx


def roi_shape(orig_shape: tuple, roi: tuple) -> tuple:
    """Find array shape after slicing ROI"""
    dummy = np.zeros((0,) + orig_shape)  # Extra 0 dim -> minimal memory use
    return dummy[np.index_exp[:] + roi].shape[1:]


class FilenameInfo:
    is_detector = False
    detector_name = None
    detector_moduleno = -1

    _rawcorr_descr = {'RAW': 'Raw', 'CORR': 'Corrected'}

    def __init__(self, path):
        self.basename = osp.basename(path)
        nameparts = self.basename[:-3].split('-')
        assert len(nameparts) == 4, self.basename
        rawcorr, runno, datasrc, segment = nameparts
        m = re.match(r'([A-Z]+)(\d+)', datasrc)

        if m and m.group(1) == 'DA':
            self.description = "Aggregated data"
        elif m and m.group(1) in DETECTOR_NAMES:
            self.is_detector = True
            name, moduleno = m.groups()
            self.detector_name = name
            self.detector_moduleno = moduleno
            self.description = "{} detector data from {} module {}".format(
                self._rawcorr_descr.get(rawcorr, '?'), name, moduleno
            )
        else:
            self.description = "Unknown data source ({})", datasrc


def find_proposal(propno):
    """Find the proposal directory for a given proposal on Maxwell"""
    if '/' in propno:
        # Already passed a proposal directory
        return propno

    t0 = time.monotonic()
    for d in iglob(osp.join(DATA_ROOT_DIR, '*/*/{}'.format(propno))):
        dt = time.monotonic() - t0
        log.info("Found proposal dir %r in %.2g s", d, dt)
        return d

    raise Exception("Couldn't find proposal dir for {!r}".format(propno))


def same_run(*args) -> bool:
    """return True if arguments objects contain data from the same RUN

    arguments can be of type *DataCollection* or *SourceData*
    """
    # DataCollection union of format version = 0.5 (no run/proposal # in
    # files) is not considered a single run.
    proposal_nos = set()
    run_nos = set()
    for dc in args:
        md = dc.run_metadata() if dc.is_single_run else {}
        proposal_nos.add(md.get("proposalNumber", -1))
        run_nos.add(md.get("runNumber", -1))

    return (len(proposal_nos) == 1 and (-1 not in proposal_nos)
            and len(run_nos) == 1 and (-1 not in run_nos))


glob_wildcards_re = re.compile(r'([*?[])')
