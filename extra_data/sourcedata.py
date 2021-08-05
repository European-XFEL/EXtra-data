from typing import List

import h5py

from .exceptions import PropertyNameError
from .file_access import FileAccess
from .keydata import KeyData

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

    def keys(self):
        """Get a set of key names for this source

        If you have used :meth:`select` to filter keys, only selected keys
        are returned.

        Only one file is used to find the keys. Within a run, all files should
        have the same keys for a given source, but if you use :meth:`union` to
        combine two runs where the source was configured differently, the
        result can be unpredictable.
        """
        if self.sel_keys is not None:
            return self.sel_keys

        # The same source may be in multiple files, but this assumes it has
        # the same keys in all files that it appears in.
        for f in self.files:
            return f.get_keys(self.source)
