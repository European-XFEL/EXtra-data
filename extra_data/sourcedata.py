import fnmatch
import re
from typing import Dict, List, Optional

import h5py
import numpy as np

from .exceptions import MultiRunError, PropertyNameError, FileStructureError
from .file_access import FileAccess
from .keydata import KeyData
from .read_machinery import (by_id, by_index, glob_wildcards_re, is_int_like,
                             same_run, select_train_ids, split_trains,
                             trains_files_index)
from .utils import isinstance_no_import


class SourceData:
    """Data for one source

    Don't create this directly; get it from ``run[source]``.
    """
    _device_class = ...
    _first_source_file = ...

    def __init__(
            self, source, *, sel_keys, train_ids, files, section,
            canonical_name, is_single_run, inc_suspect_trains=True,
    ):
        self.source = source
        self.sel_keys = sel_keys
        self.train_ids = train_ids
        self.files: List[FileAccess] = files
        self.section = section
        self.canonical_name = canonical_name
        self.is_single_run = is_single_run
        self.inc_suspect_trains = inc_suspect_trains

    def __repr__(self):
        return f"<extra_data.SourceData source={self.source!r} " \
               f"for {len(self.train_ids)} trains>"

    @property
    def is_control(self):
        """Whether this source is a control source."""
        return self.section == 'CONTROL'

    @property
    def is_instrument(self):
        """Whether this source is an instrument source."""
        return self.section == 'INSTRUMENT'

    @property
    def is_legacy(self):
        """Whether this source is a legacy name for another source."""
        return self.canonical_name != self.source

    @property
    def is_run_only(self):
        """Whether this source only has RUN keys."""
        return self.is_control and self.one_key() is None

    def _has_exact_key(self, key):
        if self.sel_keys is not None:
            return key in self.sel_keys

        for f in self.files:
            return f.has_source_key(self.source, key)

    def __contains__(self, key):
        res = self._has_exact_key(key)
        if (not res) and self.is_control:
            res = self._has_exact_key(key + '.value')
        return res

    __iter__ = None  # Disable iteration

    def __getitem__(self, key):
        if (
            isinstance(key, (by_id, by_index, list, np.ndarray, slice)) or
            isinstance_no_import(key, 'xarray', 'DataArray') or
            is_int_like(key)
        ):
            return self.select_trains(key)
        elif not isinstance(key, str):
            raise TypeError('Expected data[key] or data[train_selection]')

        if key not in self:
            raise PropertyNameError(key, self.source)
        ds0 = self.files[0].file[
            f"{self.section}/{self.source}/{key.replace('.', '/')}"
        ]
        if isinstance(ds0, h5py.Group):
            # This can only occur with a CONTROL key missing its .value suffix
            ds0 = ds0['value']
            key += '.value'

        return KeyData(
            self.source,
            key,
            train_ids=self.train_ids,
            files=self.files,
            section=self.section,
            dtype=ds0.dtype,
            eshape=ds0.shape[1:],
            inc_suspect_trains=self.inc_suspect_trains,
        )

    def _ipython_key_completions_(self):
        return list(self.keys(inc_timestamps=False))

    def _get_first_source_file(self):
        try:
            # May throw TypeError if there are only RUN keys.
            first_kd = self[self.one_key()]

            # May throw IndexError if no trains are selected.
            sample_path = first_kd.source_file_paths[0]

        except TypeError:
            if not self.is_control:
                raise FileStructureError(f'No keys present for non-CONTROL '
                                         f'source {source}')

            for file in self.files:
                link = file.file['RUN'].get(self.source, None, getlink=True)
                if link is not None:
                    break
            else:
                raise FileStructureError(f'No CONTROL or RUN keys present '
                                         f'for {source}')

            if isinstance(link, h5py.ExternalLink):
                sample_path = link.filename
            else:
                sample_path = self.files[0].filename

        except IndexError:
            sample_path = first_kd.files[0].filename

        return FileAccess(sample_path)

    @property
    def storage_class(self):
        if self._first_source_file is ...:
            self._first_source_file = self._get_first_source_file()

        return self._first_source_file.storage_class

    @property
    def data_category(self):
        if self._first_source_file is ...:
            self._first_source_file = self._get_first_source_file()

        return self._first_source_file.data_category

    @property
    def aggregator(self):
        if self._first_source_file is ...:
            self._first_source_file = self._get_first_source_file()

        return self._first_source_file.aggregator

    def keys(self, inc_timestamps=True):
        """Get a set of key names for this source

        If you have used :meth:`select` to filter keys, only selected keys
        are returned.

        For control sources, each Karabo property is stored in the file as two
        keys, with '.value' and '.timestamp' suffixes. By default, these are
        given separately. Pass ``inc_timestamps=False`` to ignore timestamps and
        drop the '.value' suffix, giving names as used in Karabo.

        Only one file is used to find the keys. Within a run, all files should
        have the same keys for a given source, but if you use :meth:`union` to
        combine two runs where the source was configured differently, the
        result can be unpredictable.
        """
        if (not inc_timestamps) and self.is_control:
            return {k[:-6] for k in self.keys() if k.endswith('.value')}

        if self.sel_keys is not None:
            return self.sel_keys

        # The same source may be in multiple files, but this assumes it has
        # the same keys in all files that it appears in.
        for f in self.files:
            return f.get_keys(self.source)

    def one_key(self, index_group=None):
        """Get a single (random) key for this source

        If you only need a single key, this can be much faster than calling
        :meth:`keys`. If *index_group* is omitted, the key may be part of
        any index group.
        """
        if self.sel_keys is not None:
            if index_group is None:
                return next(iter(self.sel_keys))

            prefix = f'{index_group}.'

            for key in self.sel_keys:
                if key.startswith(prefix):
                    return key

            raise ValueError(f'none of the selected keys is part of '
                             f'`{index_group}`')

        for f in self.files:
            return f.get_one_key(self.source, index_group)

    @property
    def index_groups(self) -> set:
        """The part of keys needed to look up index data."""
        if self.is_instrument:
            # For INSTRUMENT sources, the INDEX is saved by
            # key group, which is the first hash component. In
            # many cases this is 'data', but not always.
            if self.sel_keys is None:
                # All keys are selected.
                return self.files[0].index_groups(self.source)
            else:
                return {key.partition('.')[0] for key in self.sel_keys}
        else:
            # CONTROL data has no key group.
            return {''}

    def _glob_keys(self, pattern: str) -> Optional[set]:
        if self.is_control and not pattern.endswith(('.value', '*')):
            pattern += '.value'

        if pattern == '*':
            # When the selection refers to all keys, make sure this
            # is restricted to the current selection of keys for
            # this source.
            matched = self.sel_keys
        elif glob_wildcards_re.search(pattern) is None:
            # Selecting a single key (no wildcards in pattern)
            # This check should be faster than scanning all keys:
            matched = {pattern} if pattern in self else set()
        else:
            key_re = re.compile(fnmatch.translate(pattern))
            matched = set(filter(key_re.match, self.keys()))

        if matched == set():
            raise PropertyNameError(pattern, self.source)

        return matched

    def select_keys(self, keys) -> 'SourceData':
        """Select a subset of the keys in this source

        *keys* is either a single key name, a set of names, or a glob pattern
        (e.g. ``beamPosition.*``) matching a subset of keys. PropertyNameError
        is matched if a specified key does not exist.

        Returns a new :class:`SourceData` object.
        """
        if isinstance(keys, str):
            keys = self._glob_keys(keys)
        elif keys:
            # If a specific set of keys is selected, make sure
            # they are all valid, adding .value as needed for CONTROl keys.
            normed_keys = set()
            for key in keys:
                if self._has_exact_key(key):
                    normed_keys.add(key)
                elif self.is_control and self._has_exact_key(key + '.value'):
                    normed_keys.add(key + '.value')
                else:
                    raise PropertyNameError(key, self.source)
                keys = normed_keys
        else:
            # Catches both an empty set and None.
            # While the public API describes an empty set to
            # refer to all keys, the internal API actually uses
            # None for this case. This method is supposed to
            # accept both cases in order to natively support
            # passing a DataCollection as the selector. To keep
            # the conditions below clearer, any non-True value
            # is converted to None.
            keys = None

        if self.sel_keys is None:
            # Current keys are unspecific - use the specified keys
            new_keys = keys
        elif keys is None:
            # Current keys are specific but new selection is not - use current
            new_keys = self.sel_keys
        else:
            # Both the new and current keys are specific: take the intersection.
            # The check above should ensure this never results in an empty set,
            # but
            new_keys = self.sel_keys & keys
            assert new_keys

        return SourceData(
            self.source,
            sel_keys=new_keys,
            train_ids=self.train_ids,
            files=self.files,
            section=self.section,
            canonical_name=self.canonical_name,
            is_single_run=self.is_single_run,
            inc_suspect_trains=self.inc_suspect_trains
        )

    def select_trains(self, trains) -> 'SourceData':
        """Select a subset of trains in this data as a new :class:`SourceData` object.
        """
        return self._only_tids(select_train_ids(self.train_ids, trains))

    def _only_tids(self, tids, files=None) -> 'SourceData':
        if files is None:
            files = [
                f for f in self.files
                if f.has_train_ids(tids, self.inc_suspect_trains)
            ]
        if not files:
            # Keep 1 file, even if 0 trains selected, to get keys, dtypes, etc.
            files = [self.files[0]]

        return SourceData(
            self.source,
            sel_keys=self.sel_keys,
            train_ids=tids,
            files=files,
            section=self.section,
            canonical_name=self.canonical_name,
            is_single_run=self.is_single_run,
            inc_suspect_trains=self.inc_suspect_trains
        )

    def drop_empty_trains(self, index_group=None):
        """Select only trains with data as a new :class:`SourceData` object.

        If *index_group* is omitted, those trains with data for any of this
        source's index groups are selected.
        """
        counts = self.data_counts(labelled=False, index_group=index_group)
        tids = np.array(self.train_ids)[counts > 0]
        return self._only_tids(list(tids))

    def split_trains(self, parts=None, trains_per_part=None):
        """Split this data into chunks with a fraction of the trains each.

        Either *parts* or *trains_per_part* must be specified.

        This returns an iterator yielding new :class:`SourceData` objects.
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

    def data_counts(self, labelled=True, index_group=None):
        """Get a count of data entries in each train.

        if *index_group* is omitted, the largest count across all index
        groups is returned for each train.

        If *labelled* is True, returns a pandas series with an index of train
        IDs. Otherwise, returns a NumPy array of counts to match ``.train_ids``.
        """

        if index_group is None:
            # Collect data counts for a sample key per index group.
            data_counts = {
                index_group: self[key].data_counts(labelled=labelled)
                for index_group in self.index_groups
                if (key := self.one_key(index_group)) is not None
            }

            if not data_counts:
                data_counts = {None: np.zeros(len(self.train_ids), dtype=int)}

            if labelled:
                import pandas as pd
                return pd.DataFrame(data_counts).max(axis=1)
            else:
                return np.stack(list(data_counts.values())).max(axis=0)

        else:
            return self[self.one_key(index_group)].data_counts(
                labelled=labelled)

    def train_id_coordinates(self, index_group=None):
        """Make an array of train IDs to use alongside data this source.

        If *index_group* is omitted, the shared train ID coordinates
        across all index groups is returned if there is one. Unlike for
        ``.data_counts()``, an exception is raised if the train ID
        coordinates (and thus data counts) differ among the index groups.
        """

        if index_group is None:
            if len(self.index_groups) > 1:
                # Verify that a common train ID coordinate exists for
                # multiple index groups. The reads necessary for this
                # operation are identical to those for the train ID
                # coordinates themselves.
                counts_per_group = np.stack([
                    self.data_counts(labelled=False, index_group=index_group)
                    for index_group in self.index_groups])

                if (counts_per_group != counts_per_group[0]).any():
                    raise ValueError('source has index groups with differing '
                                     'data counts')

            index_group = self.index_groups.pop()

        return self[self.one_key(index_group)].train_id_coordinates()

    def run_metadata(self) -> Dict:
        """Get a dictionary of metadata about the run

        From file format version 1.0, the files capture: creationDate,
        daqLibrary, dataFormatVersion, karaboFramework, proposalNumber,
        runNumber, sequenceNumber, updateDate.
        """
        if not self.is_single_run:
            raise MultiRunError()

        return self.files[0].metadata()

    def run_value(self, key, *, allow_multi_run=False):
        """Get a single value from the RUN section of data files.

        This method is intended for use with data from a single run. If you
        combine data from multiple runs, it will raise MultiRunError.

        Returns the RUN parameter value corresponding to the *key* argument.
        """
        if not (self.is_single_run or allow_multi_run):
            raise MultiRunError()

        if self.is_instrument:
            raise ValueError('Only CONTROL sources have run values, '
                             f'{self.source} is an INSTRUMENT source')

        # Arbitrary file - should be the same across a run
        ds = self.files[0].file['RUN'][self.source].get(key.replace('.', '/'))
        if isinstance(ds, h5py.Group):
            # Allow for the .value suffix being omitted
            ds = ds.get('value')
        if not isinstance(ds, h5py.Dataset):
            raise PropertyNameError(key, self.source)

        val = ds[0]
        if isinstance(val, bytes):  # bytes -> str
            return val.decode('utf-8', 'surrogateescape')
        return val

    def run_values(self, inc_timestamps=True):
        """Get a dict of all RUN values for this source

        This includes keys which are also in CONTROL.
        """
        if not self.is_single_run:
            raise MultiRunError()

        if self.is_instrument:
            raise ValueError('Only CONTROL sources have run values, '
                             f'{self.source} is an INSTRUMENT source')

        res = {}
        def visitor(path, obj):
            if isinstance(obj, h5py.Dataset):
                val = obj[0]
                if isinstance(val, bytes):
                    val = val.decode('utf-8', 'surrogateescape')
                res[path.replace('/', '.')] = val

        # Arbitrary file - should be the same across a run
        self.files[0].file['RUN'][self.source].visititems(visitor)
        if not inc_timestamps:
            return {k[:-6]: v for (k, v) in res.items() if k.endswith('.value')}
        return res

    @property
    def device_class(self):
        """The name of the Karabo device class which this source belongs to

        Only for CONTROL data. This will be None for INSTRUMENT data, or if it's not available in the files.
        """
        if self._device_class is ...:
            try:
                self._device_class = self.run_value('classId', allow_multi_run=True)
            except (PropertyNameError, ValueError):
                self._device_class = None
        return self._device_class

    def union(self, *others) -> 'SourceData':
        """Combine two or more ``SourceData`` objects

        These must be for the same source, e.g. from separate runs.
        """
        if len({sd.source for sd in (self,) + others}) > 1:
            raise ValueError("Cannot use SourceData.union() with different sources")
        keygroups = [sd.sel_keys for sd in (self,) + others]
        files = set(self.files)
        train_ids = set(self.train_ids)
        for other in others:
            files.update(other.files)
            train_ids.update(other.train_ids)
        return SourceData(
            self.source,
            sel_keys=None if (None in keygroups) else set().union(*keygroups),
            train_ids=sorted(train_ids),
            files=sorted(files, key=lambda f: f.filename),
            section=self.section,
            canonical_name=self.canonical_name,
            is_single_run=same_run(self, *others),
            inc_suspect_trains=self.inc_suspect_trains
        )

    def __or__(self, other):
        return self.union(other)

    def __ior__(self, other):
        return self.union(other)
