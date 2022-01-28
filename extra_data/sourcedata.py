from copy import copy
import fnmatch
import re
from typing import Optional, List

import h5py

from .exceptions import PropertyNameError
from .file_access import FileAccess
from .keydata import KeyData
from .read_machinery import glob_wildcards_re, select_train_ids

class SourceData:
    """Data for one key in one source

    Don't create this directly; get it from ``run[source]``.
    """
    def __init__(
            self, source, *, sel_keys, train_ids, files, section,
            inc_suspect_trains=True,
    ):
        self.source = source
        self.sel_keys = sel_keys
        self.train_ids = train_ids
        self.files: List[FileAccess] = files
        self.section = section
        self.inc_suspect_trains = inc_suspect_trains

    def __repr__(self):
        return f"<extra_data.SourceData source={self.source!r} " \
               f"for {len(self.train_ids)} trains>"

    def _has_exact_key(self, key):
        if self.sel_keys is not None:
            return key in self.sel_keys

        for f in self.files:
            return f.has_source_key(self.source, key)

    def __contains__(self, key):
        res = self._has_exact_key(key)
        if (not res) and (self.section == 'CONTROL'):
            res = self._has_exact_key(key + '.value')
        return res

    def __getitem__(self, key):
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
        if (not inc_timestamps) and (self.section == 'CONTROL'):
            return {k[:-6] for k in self.keys() if k.endswith('.value')}

        if self.sel_keys is not None:
            return self.sel_keys

        # The same source may be in multiple files, but this assumes it has
        # the same keys in all files that it appears in.
        for f in self.files:
            return f.get_keys(self.source)

    def _index_group_names(self) -> set:
        if self.section == 'INSTRUMENT':
            # For INSTRUMENT sources, the INDEX is saved by
            # key group, which is the first hash component. In
            # many cases this is 'data', but not always.
            if self.sel_keys is None:
                # All keys are selected.
                return self.files[0].index_group_names(self.source)
            else:
                return {key.partition('.')[0] for key in self.sel_keys}
        else:
            # CONTROL data has no key group.
            return {''}

    def _glob_keys(self, pattern: str) -> Optional[set]:
        if self.section == 'CONTROL' and not pattern.endswith(('.value', '*')):
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
        if isinstance(keys, str):
            keys = self._glob_keys(keys)
        elif keys:
            # If a specific set of keys is selected, make sure
            # they are all valid.
            for key in keys:
                if key not in self:
                    raise PropertyNameError(key, self.source)
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

        res = copy(self)
        res.sel_keys = new_keys
        return res

    def select_trains(self, trains) -> 'SourceData':
        return self._only_tids(select_train_ids(self.train_ids, trains))

    def _only_tids(self, tids) -> 'SourceData':
        res = copy(self)
        res.train_ids = tids
        res.files = [
            f for f in self.files
            if f.has_train_ids(tids, self.inc_suspect_trains)
        ] or [self.files[0]]
        # ^ Keep 1 file, even if 0 trains selected, to get keys, datatypes, etc.

        return res

    def union(self, *others) -> 'SourceData':
        keygroups = [sd.sel_keys for sd in (self,) + others]
        files = set(self.files)
        train_ids = set(self.train_ids)
        for other in others:
            files.update(other.files)
            train_ids.update(other.train_ids)
        res = copy(self)
        res.sel_keys = None if (None in keygroups) else set().union(*keygroups)
        res.files = sorted(files, key=lambda f: f.filename)
        res.train_ids = sorted(train_ids)
        return res
