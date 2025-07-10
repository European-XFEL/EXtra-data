# coding: utf-8
"""
Collection of classes and functions to help reading HDF5 file generated at
The European XFEL.

Copyright (c) 2017, European X-Ray Free-Electron Laser Facility GmbH
All rights reserved.

You should have received a copy of the 3-Clause BSD License along with this
program. If not, see <https://opensource.org/licenses/BSD-3-Clause>
"""

import datetime
import fnmatch
import logging
import os
import os.path as osp
import re
import signal
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from itertools import groupby, chain
from multiprocessing import Pool
from operator import index
from pathlib import Path
from typing import Tuple
from warnings import warn

import h5py
import numpy as np

from . import locality, voview
from .aliases import AliasIndexer
from .exceptions import (MultiRunError, PropertyNameError, SourceNameError,
                         TrainIDError)
from .file_access import FileAccess
from .keydata import KeyData
from .read_machinery import (DETECTOR_SOURCE_RE, by_id, by_index,
                             find_proposal, glob_wildcards_re, is_int_like,
                             same_run, select_train_ids)
from .run_files_map import RunFilesMap
from .sourcedata import SourceData
from .utils import available_cpu_cores, isinstance_no_import

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

def ignore_sigint():
    # Used in child processes to prevent them from receiving KeyboardInterrupt
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class DataCollection:
    """An assemblage of data generated at European XFEL

    Data consists of *sources* which each have *keys*. It is further
    organised by *trains*, which are identified by train IDs.

    You normally get an instance of this class by calling :func:`H5File`
    for a single file or :func:`RunDirectory` for a directory.
    """
    def __init__(
            self, files, sources_data=None, train_ids=None, aliases=None,
            ctx_closes=False, *, inc_suspect_trains=True, is_single_run=False,
            alias_files=None
    ):
        self.files = list(files)
        self.ctx_closes = ctx_closes
        self.inc_suspect_trains = inc_suspect_trains
        self.is_single_run = is_single_run

        if train_ids is None:
            if inc_suspect_trains:
                tid_sets = [f.train_ids for f in files]
            else:
                tid_sets = [f.valid_train_ids for f in files]
            train_ids = sorted(set().union(*tid_sets))
        self.train_ids = train_ids

        if sources_data is None:
            files_by_sources = defaultdict(list)
            legacy_sources = dict()
            for f in self.files:
                for source in f.control_sources:
                    files_by_sources[source, 'CONTROL'].append(f)
                for source in f.instrument_sources:
                    files_by_sources[source, 'INSTRUMENT'].append(f)
                legacy_sources.update(f.legacy_sources)
            sources_data = {
                src: SourceData(src,
                    sel_keys=None,
                    train_ids=train_ids,
                    files=files,
                    section=section,
                    canonical_name=legacy_sources.get(src, src),
                    is_single_run=self.is_single_run,
                    inc_suspect_trains=self.inc_suspect_trains
                )
                for ((src, section), files) in files_by_sources.items()
            }
        self._sources_data = sources_data

        # Note that _alias_files is only for tracking where the aliases came
        # from, the actual aliases are stored in _aliases.
        self._alias_files = [] if alias_files is None else alias_files
        self._aliases = aliases or {}
        self.alias = AliasIndexer(self)

        # Throw an error if we have conflicting data for the same source
        self._check_source_conflicts()

        self.control_sources = frozenset({
            name for (name, sd) in self._sources_data.items()
            if sd.section == 'CONTROL'
        })
        self.instrument_sources = frozenset({
            name for (name, sd) in self._sources_data.items()
            if sd.section == 'INSTRUMENT'
        })
        self.legacy_sources = {
            name: sd.canonical_name for (name, sd)
            in self._sources_data.items() if sd.is_legacy
        }

    @staticmethod
    def _open_file(path, cache_info=None):
        try:
            fa = FileAccess(path, _cache_info=cache_info)
        except Exception as e:
            return osp.basename(path), str(e)
        else:
            return osp.basename(path), fa

    @classmethod
    def from_paths(
            cls, paths, _files_map=None, *, inc_suspect_trains=True,
            is_single_run=False, parallelize=True
    ):
        files = []
        uncached = []

        def handle_open_file_attempt(fname, fa):
            if isinstance(fa, FileAccess):
                files.append(fa)
            else:
                print(f"Skipping file {fname}", file=sys.stderr)
                print(f"  (error was: {fa})", file=sys.stderr)

        for path in paths:
            cache_info = _files_map and _files_map.get(path)
            if cache_info and ('flag' in cache_info):
                filename, fa = cls._open_file(path, cache_info=cache_info)
                handle_open_file_attempt(filename, fa)
            else:
                uncached.append(path)

        if uncached:
            # Open the files either in parallel or serially
            if parallelize:
                nproc = min(available_cpu_cores(), len(uncached))
                with Pool(processes=nproc, initializer=ignore_sigint) as pool:
                    for fname, fa in pool.imap_unordered(cls._open_file, uncached):
                        handle_open_file_attempt(fname, fa)
            else:
                for path in uncached:
                    handle_open_file_attempt(*cls._open_file(path))

        if not files:
            raise Exception("All HDF5 files specified are unusable")

        return cls(
            files, ctx_closes=True, inc_suspect_trains=inc_suspect_trains,
            is_single_run=is_single_run,
        )

    @classmethod
    def from_path(cls, path, *, inc_suspect_trains=True):
        files = [FileAccess(path)]
        return cls(
            files, ctx_closes=True, inc_suspect_trains=inc_suspect_trains,
            is_single_run=True
        )

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
    def selection(self):
        # This was previously a regular attribute, which code may have relied on.
        return {src: srcdata.sel_keys for src, srcdata in self._sources_data.items()}

    @property
    def _source_index(self):
        warn(
            "DataCollection._source_index will be removed. "
            "Contact da-support@xfel.eu if you need to discuss alternatives.",
            stacklevel=2
        )
        return {src: srcdata.files for src, srcdata in self._sources_data.items()}

    @property
    def all_sources(self):
        return self.control_sources | self.instrument_sources

    @property
    def detector_sources(self):
        return set(filter(DETECTOR_SOURCE_RE.match, self.instrument_sources)) \
            - self.legacy_sources.keys()

    def _check_field(self, source, key):
        if source not in self.all_sources:
            raise SourceNameError(source)

        if key not in self[source]:
            raise PropertyNameError(key, source)

    def keys_for_source(self, source):
        """Get a set of key names for the given source

        If you have used :meth:`select` to filter keys, only selected keys
        are returned.

        Only one file is used to find the keys. Within a run, all files should
        have the same keys for a given source, but if you use :meth:`union` to
        combine two runs where the source was configured differently, the
        result can be unpredictable.
        """
        return self._get_source_data(source).keys()

    # Leave old name in case anything external was using it:
    _keys_for_source = keys_for_source

    def _get_key_data(self, source, key):
        return self._get_source_data(source)[key]

    def _get_source_data(self, source):
        if source not in self._sources_data:
            raise SourceNameError(source)

        sd = self._sources_data[source]

        if sd.is_legacy:
            warn(f"{source} is a legacy name for {self.legacy_sources[source]}. "
                 f"Access via this name will be removed for future data.",
                 DeprecationWarning,
                 stacklevel=3)

        return self._sources_data[source]

    def __getitem__(self, item):
        if (
            isinstance(item, tuple) and
            len(item) == 2 and
            all(isinstance(e, str) for e in item)
        ):
            return self._get_key_data(*item)
        elif isinstance(item, str):
            return self._get_source_data(item)
        elif (
            isinstance(item, (by_id, by_index, list, np.ndarray, slice)) or
            isinstance_no_import(item, 'xarray', 'DataArray') or
            is_int_like(item)
        ):
            return self.select_trains(item)

        raise TypeError("Expected data[source], data[source, key] or data[train_selection]")

    def __contains__(self, item):
        if (
            isinstance(item, tuple) and
            len(item) == 2 and
            all(isinstance(e, str) for e in item)
        ):
            return item[0] in self.all_sources and \
                item[1] in self._get_source_data(item[0])
        elif isinstance(item, str):
            return item in self.all_sources

        return False

    __iter__ = None  # Disable iteration

    def _ipython_key_completions_(self):
        return list(self.all_sources)

    def get_entry_shape(self, source, key):
        """Get the shape of a single data entry for the given source & key"""
        return self._get_key_data(source, key).entry_shape

    def get_dtype(self, source, key):
        """Get the numpy data type for the given source & key"""
        return self._get_key_data(source, key).dtype

    def _check_data_missing(self, tid) -> bool:
        """Return True if a train does not have data for all sources"""
        for source in self.control_sources:
            file, _ = self._find_data(source, tid)
            if file is None:
                return True

        # No need to evaluate this for legacy sources as well.
        for source in self.instrument_sources - self.legacy_sources.keys():
            file, pos = self._find_data(source, tid)
            if file is None:
                return True

            groups = {k.partition('.')[0] for k in self.keys_for_source(source)}
            for group in groups:
                _, counts = file.get_index(source, group)
                if counts[pos] == 0:
                    return True

        return False

    def trains(self, devices=None, train_range=None, *, require_all=False,
               flat_keys=False, keep_dims=False):
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

        flat_keys: bool
            False (default) returns nested dictionaries in each
            iteration indexed by source and then key. True returns a
            flat dictionary indexed by (source, key) tuples.

        keep_dims: bool
            False (default) drops the first dimension when there is
            a single entry. True preserves this dimension.

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
        return iter(TrainIterator(dc, require_all=require_all,
                                  flat_keys=flat_keys, keep_dims=keep_dims))

    def train_from_id(
        self, train_id, devices=None, *, flat_keys=False, keep_dims=False):
        """Get train data for specified train ID.

        Parameters
        ----------

        train_id: int
            The train ID
        devices: dict or list, optional
            Filter data by sources and keys.
            Refer to :meth:`select` for how to use this.
        flat_keys: bool
            False (default) returns a nested dict indexed by source and then key.
            True returns a flat dictionary indexed by (source, key) tuples.
        keep_dims: bool
            False (default) drops the first dimension when there is
            a single entry. True preserves this dimension.

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

            firsts, counts = file.get_index(source, '')
            first, count = firsts[pos], counts[pos]
            if not count:
                continue

            for key in self.keys_for_source(source):
                path = '/CONTROL/{}/{}'.format(source, key.replace('.', '/'))
                source_data[key] = file.file[path][first]

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
                if count == 1 and not keep_dims:
                    source_data[key] = file.file[path][first]
                else:
                    source_data[key] = file.file[path][first : first + count]

        if flat_keys:
            # {src: {key: data}} -> {(src, key): data}
            res = {(src, key): v for src, source_data in res.items()
                   for (key, v) in source_data.items()}

        return train_id, res

    def train_from_index(
        self, train_index, devices=None, *, flat_keys=False, keep_dims=False):
        """Get train data of the nth train in this data.

        Parameters
        ----------
        train_index: int
            Index of the train in the file.
        devices: dict or list, optional
            Filter data by sources and keys.
            Refer to :meth:`select` for how to use this.
        flat_keys: bool
            False (default) returns a nested dict indexed by source and then key.
            True returns a flat dictionary indexed by (source, key) tuples.
        keep_dims: bool
            False (default) drops the first dimension when there is
            a single entry. True preserves this dimension.

        Returns
        -------

        tid : int
            The train ID of the returned train
        data : dict
            The data for this train, keyed by device name
        """
        train_id = self.train_ids[train_index]
        return self.train_from_id(
            int(train_id), devices=devices,
            flat_keys=flat_keys, keep_dims=keep_dims)

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
        return self._get_key_data(source, key).data_counts()

    def get_series(self, source, key):
        """Return a pandas Series for a 1D data field defined by source & key.

        See :meth:`.KeyData.series` for details.
        """
        return self._get_key_data(source, key).series()

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

    def get_array(self, source, key, extra_dims=None, roi=(), name=None):
        """Return a labelled array for a data field defined by source and key.

        see :meth:`.KeyData.xarray` for details.
        """
        if isinstance(roi, by_index):
            roi = roi.value

        return self._get_key_data(source, key).xarray(
            extra_dims=extra_dims, roi=roi, name=name)

    def get_dask_array(self, source, key, labelled=False):
        """Get a Dask array for a data field defined by source and key.

        see :meth:`.KeyData.dask_array` for details.
        """
        return self._get_key_data(source, key).dask_array(labelled=labelled)

    def get_run_value(self, source, key):
        """Get a single value from the RUN section of data files.

        RUN records each property of control devices as a snapshot at the
        beginning of the run. This includes properties which are not saved
        continuously in CONTROL data.

        This method is intended for use with data from a single run. If you
        combine data from multiple runs, it will raise MultiRunError.

        Parameters
        ----------

        source: str
            Control device name, e.g. "HED_OPT_PAM/CAM/SAMPLE_CAM_4".
        key: str
            Key of parameter within that device, e.g. "triggerMode".
        """
        return self._get_source_data(source).run_value(key)

    def get_run_values(self, source) -> dict:
        """Get a dict of all RUN values for the given source

        This includes keys which are also in CONTROL.

        Parameters
        ----------

        source: str
            Control device name, e.g. "HED_OPT_PAM/CAM/SAMPLE_CAM_4".
        """
        return self._get_source_data(source).run_values()

    def _merge_aliases(self, alias_dicts):
        """Merge multiple alias dictionaries and check for conflicts."""

        new_aliases = {}

        for aliases in alias_dicts:
            for alias, literal in aliases.items():
                alias = alias.lower().replace('_', '-')
                if new_aliases.setdefault(alias, literal) != literal:
                    raise ValueError(f'conflicting alias definition '
                                     f'for {alias} (or {alias.upper()}, '
                                     f'{alias.replace("-", "_")}, etc.)')

        return new_aliases

    def _merge_alias_files(self, *alias_files):
        all_files = chain.from_iterable(alias_files)
        return sorted(set(all_files))

    def union(self, *others):
        """Join the data in this collection with one or more others.

        This can be used to join multiple sources for the same trains,
        or to extend the same sources with data for further trains.
        The order of the datasets doesn't matter. Any aliases defined on
        the collections are combined as well unless their values conflict.

        Note that the trains for each source are unioned as well, such that
        ``run.train_ids == run[src].train_ids``.

        Returns a new :class:`DataCollection` object.
        """

        sources_data_multi = defaultdict(list)
        for dc in (self,) + others:
            for source, srcdata in dc._sources_data.items():
                sources_data_multi[source].append(srcdata)

        sources_data = {src: src_datas[0].union(*src_datas[1:])
                        for src, src_datas in sources_data_multi.items()}

        aliases = self._merge_aliases(
            [self._aliases] + [dc._aliases for dc in others])
        alias_files = self._merge_alias_files(self._alias_files,
                                              *[dc._alias_files for dc in others])

        train_ids = sorted(set().union(*[sd.train_ids for sd in sources_data.values()]))
        # Update the internal list of train IDs for the sources
        for sd in sources_data.values():
            sd.train_ids = train_ids

        files = set().union(*[sd.files for sd in sources_data.values()])

        return DataCollection(
            files, sources_data=sources_data, train_ids=train_ids,
            aliases=aliases, inc_suspect_trains=self.inc_suspect_trains,
            is_single_run=same_run(self, *others), alias_files=alias_files
        )

    def __or__(self, other):
        return self.union(other)

    def __ior__(self, other):
        return self.union(other)

    def _parse_aliases(self, alias_defs):
        """Parse alias definitions into alias dictionaries."""

        alias_dicts = []
        alias_files = []

        def is_valid_alias(k, v):
            return (isinstance(k, str) and (
                isinstance(v, str) or (isinstance(v, tuple) and len(v) == 2)
            ))

        for alias_def in alias_defs:
            if isinstance(alias_def, Mapping):
                if any([not is_valid_alias(k, v) for k, v in alias_def.items()]):
                    raise ValueError('alias definition by dict must be all '
                                     'str keys to str values for sources or '
                                     '2-len tuples for sourcekeys')

                alias_dicts.append(alias_def)
            elif isinstance(alias_def, (str, os.PathLike)):
                # From a file.
                alias_files.append(Path(alias_def))
                alias_dicts.append(
                    self._load_aliases_from_file(Path(alias_def)))

        return alias_dicts, alias_files

    def _load_aliases_from_file(self, aliases_path):
        """Load alias definitions from file."""

        if aliases_path.suffix == '.json':
            import json

            with open(aliases_path, 'r') as f:
                data = json.load(f)

        elif aliases_path.suffix in ['.yaml', '.yml']:
            import yaml

            with open(aliases_path, 'r') as f:
                data = yaml.safe_load(f)

        elif aliases_path.suffix == '.toml':
            try:
                from tomli import load as load_toml
            except ImportError:
                # Try the built-in tomllib for 3.11+.
                from tomllib import load as load_toml

            with open(aliases_path, 'rb') as f:
                data = load_toml(f)

        aliases = {}

        def walk_dict_value(source, key_aliases):
            for alias, key in key_aliases.items():
                aliases[alias] = (source, key)

        for key, value in data.items():
            if isinstance(value, str):
                # Source alias.
                aliases[key] = value
            elif isinstance(value, list) and len(value) == 2:
                # Sourcekey alias by explicit list.
                aliases[key] = tuple((str(x) for x in value))
            elif isinstance(value, dict):
                # Sourcekey alias by nested mapping.
                walk_dict_value(key, value)
            else:
                raise ValueError(f"unsupported literal type for alias '{key}'")

        return aliases

    def with_aliases(self, *alias_defs):
        """Apply aliases for convenient source and key access.

        Allows to define aliases for sources or source-key combinations
        that may be used instead of their literal names to retrieve
        :class:`SourceData` and :class:`KeyData` objects via
        :attr:`.DataCollection.alias`.

        Multiple alias definitions may be passed as positional arguments
        in different formats:

        1. Passing a dictionary mapping aliases to sources (passed as strings)
           or source-key pairs (passed as a 2-len tuple of strings).

        2. Passing a string or PathLike pointing to a JSON, YAML (requires
           pyYAML installed) or TOML (requires Python 3.11 or with tomli
           installed) file containing the aliases. For unsupported formats,
           an :class:`ImportError` is raised.

           The file should contain mappings from alias to sources as strings
           or source-key pairs as lists. In addition, source-key aliases may
           be defined by nested key-value pairs according to the respective
           format, shown here in YAML:

           .. code-block:: yaml

             # Source alias.
             sa1-xgm: SA1_XTD2_XGM/XGM/DOOCS

             # Direct source key alias.
             sa1-intensity: [SA1_XTD2_XGM/XGM/DOOCS:output, data.intensityTD]

             # Nested source key alias, useful if you want aliases for multiple
             # keys of the same source.
             SA3_XTD10_MONO/MDL/PHOTON_ENERGY:
               mono-central-energy: actualEnergy

        Returns a new :class:`DataCollection` object with the aliases
        for sources and keys.
        """

        # Check for conflicts within these definitions
        new_aliases, new_alias_files = self._parse_aliases(alias_defs)
        new_aliases = self._merge_aliases([self._aliases] + new_aliases)
        alias_files = self._merge_alias_files(self._alias_files, new_alias_files)

        return DataCollection(
            self.files, sources_data=self._sources_data,
            train_ids=self.train_ids, aliases=new_aliases,
            inc_suspect_trains=self.inc_suspect_trains,
            is_single_run=self.is_single_run, alias_files=alias_files
        )

    def only_aliases(self, *alias_defs, strict=False, require_all=False):
        """Apply aliases and select only the aliased sources and keys.

        A convenient function around :meth:`DataCollection.with_aliases`
        and :meth:`DataCollection.select` applying both the aliases passed
        as ``alias_defs`` to the former and then selecting down the
        :class:`DataCollection` to any sources and/or their keys for which
        aliases exist.

        By default and unlike :meth:`DataCollection.select`, any sources
        or keys present in the alias definitions but not the data itself are
        ignored. This can be changed via the optional argument ``strict``.

        The optional ``require_all`` argument restricts the trains to those for
        which all selected sources and keys have at least one data entry. By
        default, all trains remain selected.

        Returns a new :class:`DataCollection` object with only the aliased
        sources and keys.
        """

        # Create new aliases.
        new_aliases, new_alias_files = self._parse_aliases(alias_defs)
        aliases = self._merge_aliases([self._aliases] + new_aliases)
        alias_files = self._merge_alias_files(self._alias_files, new_alias_files)

        # Set of sources aliased.
        aliased_sources = {literal for literal in aliases.values()
                           if isinstance(literal, str)}

        # In the current implementation of DataCollection.select(), any
        # occurence of a wildcard glob will include all keys for a given
        # source, even if specific keys are listed as well. To be safe,
        # the source aliases are picked first and no specific sourcekey
        # aliases for the same source are included in the selection.

        # Entire source selections.
        selection = [(source, '*') for source in aliased_sources]

        # Specific key selections.
        selection += [
            literal for literal in aliases.values()
            if isinstance(literal, tuple) \
                and literal[0] not in aliased_sources
        ]

        if not strict:
            # If strict mode is disabled, any non-existing sources or
            # keys are stripped out.

            existing_sel_idx = []

            for sel_idx, (source, key) in enumerate(selection):
                try:
                    sd = self[source]
                except SourceNameError:
                    # Source not present.
                    continue
                else:
                    if key != '*' and key not in sd:
                        # Source present, but not key.
                        continue

                existing_sel_idx.append(sel_idx)

            selection = [selection[sel_idx] for sel_idx in existing_sel_idx]

        # Create a new DataCollection from selecting and add the aliases.
        new_data = self.select(selection, require_all=require_all)
        new_data._aliases = aliases
        new_data._alias_files = alias_files

        return new_data

    def drop_aliases(self):
        """Return a new DataCollection without any aliases."""

        return DataCollection(
            self.files, sources_data=self._sources_data,
            train_ids=self.train_ids, aliases={},
            inc_suspect_trains=self.inc_suspect_trains,
            is_single_run=self.is_single_run
        )

    def _expand_selection(self, selection):
        if isinstance(selection, dict):
            # {source: {key1, key2}}
            # {source: set()} or {source: None} -> all keys for this source

            res = {}
            for source, in_keys in selection.items():
                if source not in self.all_sources:
                    raise SourceNameError(source)

                # Empty dict was accidentally allowed and tested; keep it
                # working just in case.
                if in_keys == {}:
                    in_keys = set()

                if in_keys is not None and not isinstance(in_keys, set):
                    raise TypeError(
                        f"keys in selection dict should be a set or None (got "
                        f"{in_keys!r})"
                    )

                res[source] = self._sources_data[source].select_keys(in_keys)

            return res

        elif isinstance(selection, Iterable):
            # selection = [('src_glob', 'key_glob'), ...]
            # OR          ['src_glob', 'src_glob', ...]
            sources_data_multi = defaultdict(list)
            for globs in selection:
                if isinstance(globs, str):
                    src_glob = globs
                    key_glob = '*'
                else:
                    src_glob, key_glob = globs
                for source, keys in self._select_glob(src_glob, key_glob).items():
                    sources_data_multi[source].append(
                        self._sources_data[source].select_keys(keys)
                    )
            return {src: src_datas[0].union(*src_datas[1:])
                    for src, src_datas in sources_data_multi.items()}
        elif isinstance(selection, DataCollection):
            return self._expand_selection(selection.selection)
        elif isinstance(selection, SourceData):
            return {selection.source: selection}
        elif isinstance(selection, KeyData):
            src = selection.source
            return {src: self._sources_data[src].select_keys({selection.key})}
        else:
            raise TypeError("Unknown selection type: {}".format(type(selection)))

    def _select_glob(self, source_glob, key_glob):
        source_re = re.compile(fnmatch.translate(source_glob))
        key_re = re.compile(fnmatch.translate(key_glob))
        if key_glob.endswith(('.value', '*')):
            ctrl_key_glob = key_glob
            ctrl_key_re = key_re
        else:
            # Add .value suffix for keys of CONTROL sources
            ctrl_key_glob = key_glob + '.value'
            ctrl_key_re = re.compile(fnmatch.translate(ctrl_key_glob))

        matched = {}
        for source in self.all_sources:
            if not source_re.match(source):
                continue

            if key_glob == '*':
                # When the selection refers to all keys, make sure this
                # is restricted to the current selection of keys for
                # this source.

                if self.selection[source] is None:
                    matched[source] = None
                else:
                    matched[source] = self.selection[source]
            elif glob_wildcards_re.search(key_glob) is None:
                # Selecting a single key (no wildcards in pattern)
                # This check should be faster than scanning all keys:
                k = ctrl_key_glob if source in self.control_sources else key_glob
                if k in self._sources_data[source]:
                    matched[source] = {k}
            else:
                r = ctrl_key_re if source in self.control_sources else key_re
                keys = set(filter(r.match, self.keys_for_source(source)))
                if keys:
                    matched[source] = keys

        if not matched:
            raise ValueError("No matches for pattern {}"
                             .format((source_glob, key_glob)))
        return matched

    def select(self, seln_or_source_glob, key_glob='*', require_all=False,
               require_any=False, *, warn_drop_trains_frac=1.):
        """Select a subset of sources and keys from this data.

        There are four possible ways to select data:

        1. With two glob patterns (see below) for source and key names::

            # Select data in the image group for any detector sources
            sel = run.select('*/DET/*', 'image.*')

        2. With an iterable of source glob patterns, or (source, key) patterns::

            # Select image.data and image.mask for any detector sources
            sel = run.select([('*/DET/*', 'image.data'), ('*/DET/*', 'image.mask')])

            # Select & align undulator & XGM devices
            sel = run.select(['*XGM/*', 'MID_XTD1_UND/DOOCS/ENERGY'], require_all=True)

           Data is included if it matches any of the pattern pairs.

        3. With a dict of source names mapped to sets of key names
           (or empty sets to get all keys)::

            # Select image.data from one detector source, and all data from one XGM
            sel = run.select({'SPB_DET_AGIPD1M-1/DET/0CH0:xtdf': {'image.data'},
                              'SA1_XTD2_XGM/XGM/DOOCS': set()})

           Unlike the others, this option *doesn't* allow glob patterns.
           It's a more precise but less convenient option for code that knows
           exactly what sources and keys it needs.

        4. With an existing DataCollection, SourceData or KeyData object::

             # Select the same data contained in another DataCollection
             prev_run.select(sel)

        The optional `require_all` and `require_any` arguments restrict the
        trains to those for which all or at least one selected sources and
        keys have at least one data entry. By default, all trains remain selected.

        With `require_all=True`, a warning will be shown if there are no trains
        with all the required data. Setting `warn_drop_trains_frac` can show the
        same warning if there are a few remaining trains. This is a number 0-1
        representing the fraction of trains dropped for one source (default 1).

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
        sources_data = self._expand_selection(seln_or_source_glob)

        if require_all or require_any:
            # Select only those trains for which all (require_all) or at
            # least one (require_any) selected sources and keys have
            # data, i.e. have a count > 0 in their respective INDEX
            # section.

            if require_all:
                train_ids = self.train_ids
            else:  # require_any
                # Empty list would be converted to np.float64 array.
                train_ids = np.empty(0, dtype=np.uint64)

            for source, srcdata in sources_data.items():
                if srcdata.is_run_only:
                    continue

                n_trains_prev = len(train_ids)
                for group in srcdata.index_groups:
                    source_tids = np.empty(0, dtype=np.uint64)

                    for f in self._sources_data[source].files:
                        valid = True if self.inc_suspect_trains else f.validity_flag
                        # Add the trains with data in each file.
                        _, counts = f.get_index(source, group)
                        source_tids = np.union1d(
                            f.train_ids[valid & (counts > 0)], source_tids
                        )

                    # Remove any trains previously selected, for which this
                    # selected source and key group has no data.

                    if require_all:
                        train_ids = np.intersect1d(train_ids, source_tids)
                    else:  # require_any
                        train_ids = np.union1d(train_ids, source_tids)

                n_drop = n_trains_prev - len(train_ids)
                if n_trains_prev and (n_drop / n_trains_prev) >= warn_drop_trains_frac:
                    warn(f"{n_drop}/{n_trains_prev} ({n_drop / n_trains_prev :.0%})"
                         f" trains dropped when filtering by {source}")

            train_ids = list(train_ids)  # Convert back to a list.
            sources_data = {
                src: srcdata._only_tids(train_ids)
                for src, srcdata in sources_data.items()
            }

        else:
            train_ids = self.train_ids

        files = set().union(*[sd.files for sd in sources_data.values()])

        return DataCollection(
            files, sources_data, train_ids=train_ids, aliases=self._aliases,
            inc_suspect_trains=self.inc_suspect_trains,
            is_single_run=self.is_single_run, alias_files=self._alias_files
        )

    def deselect(self, seln_or_source_glob, key_glob='*'):
        """Select everything except the specified sources and keys.

        This takes the same arguments as :meth:`select`, but the sources and
        keys you specify are dropped from the selection.

        Returns a new :class:`DataCollection` object for the remaining data.
        """

        if isinstance(seln_or_source_glob, str):
            seln_or_source_glob = [(seln_or_source_glob, key_glob)]
        deselection = self._expand_selection(seln_or_source_glob)

        # Subtract deselection from selection on self
        sources_data = {}
        for source, srcdata in self._sources_data.items():
            if source not in deselection:
                sources_data[source] = srcdata
                continue

            desel_keys = deselection[source].sel_keys
            if desel_keys is None:
                continue  # Drop the entire source

            remaining_keys = srcdata.keys() - desel_keys

            if remaining_keys:
                sources_data[source] = srcdata.select_keys(remaining_keys)

        files = set().union(*[sd.files for sd in sources_data.values()])

        return DataCollection(
            files, sources_data=sources_data, train_ids=self.train_ids,
            aliases=self._aliases, inc_suspect_trains=self.inc_suspect_trains,
            is_single_run=self.is_single_run, alias_files=self._alias_files
        )

    def select_trains(self, train_range):
        """Select a subset of trains from this data.

        Slice trains by position within this data::

            sel = run.select_trains(np.s_[:5])

        Or select trains by train ID, with a slice or a list::

            from extra_data import by_id
            sel1 = run.select_trains(by_id[142844490 : 142844495])
            sel2 = run.select_trains(by_id[[142844490, 142844493, 142844494]])

        Returns a new :class:`DataCollection` object for the selected trains.

        Raises
        ------
        ValueError
            If given train IDs do not overlap with the trains in this data.
        """
        new_train_ids = select_train_ids(self.train_ids, train_range)

        sources_data = {
            src: srcdata._only_tids(new_train_ids)
            for src, srcdata in self._sources_data.items()
        }

        files = set().union(*[sd.files for sd in sources_data.values()])

        return DataCollection(
            files, sources_data=sources_data, train_ids=new_train_ids,
            aliases=self._aliases, inc_suspect_trains=self.inc_suspect_trains,
            is_single_run=self.is_single_run, alias_files=self._alias_files
        )

    def split_trains(self, parts=None, trains_per_part=None):
        """Split this data into chunks with a fraction of the trains each.

        Either *parts* or *trains_per_part* must be specified.

        This returns an iterator yielding new :class:`DataCollection` objects.
        The parts will have similar sizes, e.g. splitting 11 trains
        with ``trains_per_part=8`` will produce 5 & 6 trains, not 8 & 3.

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
        for source in self._sources_data.values():
            assert source.train_ids == self.train_ids

        def dict_zip(iter_d):
            while True:
                try:
                    yield {k: next(v) for (k, v) in iter_d.items()}
                except StopIteration:
                    return

        for sources_data_part in dict_zip({
            n: s.split_trains(parts=parts, trains_per_part=trains_per_part)
            for (n, s) in self._sources_data.items()
        }):
            files = set().union(*[sd.files for sd in sources_data_part.values()])
            train_ids = list(sources_data_part.values())[0].train_ids

            yield DataCollection(
                files, sources_data=sources_data_part, train_ids=train_ids,
                aliases=self._aliases, inc_suspect_trains=self.inc_suspect_trains,
                is_single_run=self.is_single_run, alias_files=self._alias_files
            )

    def _check_source_conflicts(self):
        """Check for data with the same source and train ID in different files.
        """
        sources_with_conflicts = set()
        files_conflict_cache = {}

        def files_have_conflict(files):
            fset = frozenset({f.filename for f in files})
            if fset not in files_conflict_cache:
                if self.inc_suspect_trains:
                    tids = np.concatenate([f.train_ids for f in files])
                else:
                    tids = np.concatenate([f.valid_train_ids for f in files])
                files_conflict_cache[fset] = len(np.unique(tids)) != len(tids)
            return files_conflict_cache[fset]

        for source, srcdata in self._sources_data.items():
            if files_have_conflict(srcdata.files):
                sources_with_conflicts.add(source)

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
        return self._get_key_data(source, key)._data_chunks

    def _find_data(self, source, train_id) -> Tuple[FileAccess, int]:
        for f in self._sources_data[source].files:
            ixs = (f.train_ids == train_id).nonzero()[0]
            if self.inc_suspect_trains and ixs.size > 0:
                return f, ixs[0]

            for ix in ixs:
                if f.validity_flag[ix]:
                    return f, ix

        return None, None

    def __repr__(self):
        return f"<extra_data.DataCollection for {len(self.all_sources)} " \
               f"sources and {len(self.train_ids)} trains>"

    def info(self, details_for_sources=(), with_aggregators=False):
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
            try:
                td = datetime.timedelta(seconds=int(seconds))
            except OverflowError:  # Can occur if a train ID is corrupted
                span_txt = "OverflowError (one or more train IDs are probably wrong)"
            else:
                span_txt = f'{td}.{int(deciseconds)}'

        # disp
        print('# of trains:   ', train_count)
        print('Duration:      ', span_txt)
        print('First train ID:', first_train)
        print('Last train ID: ', last_train)
        print()

        if not details_for_sources:
            # Include summary section for multi-module detectors unless
            # source details are enabled.

            sources_by_detector = {}
            for source in self.detector_sources:
                name, modno = DETECTOR_SOURCE_RE.match(source).groups((1, 2))
                sources_by_detector.setdefault(name, {})[modno] = source

            for detector_name in sorted(sources_by_detector.keys()):
                detector_modules = sources_by_detector[detector_name]

                print("{} XTDF detector modules of {}/*".format(
                    len(detector_modules), detector_name
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

        # Invert aliases for faster lookup.
        src_aliases = defaultdict(set)
        srckey_aliases = defaultdict(lambda: defaultdict(set))

        for alias, literal in self._aliases.items():
            if isinstance(literal, str):
                src_aliases[literal].add(alias)
            else:
                srckey_aliases[literal[0]][literal[1]].add(alias)

        def src_alias_list(s):
            if src_aliases[s]:
                alias_str = ', '.join(src_aliases[s])
                return f'<{alias_str}>'
            return ''

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

                if k in srckey_aliases[s]:
                    alias_str = ' <' + ', '.join(srckey_aliases[s][k]) + '>'
                else:
                    alias_str = ''

                print(f"{prefix}{k}{alias_str}\t[{dt}{entry_info}]")

        if details_for_sources:
            # All instrument sources with details enabled.
            displayed_inst_srcs = self.instrument_sources - self.legacy_sources.keys()
            print(len(displayed_inst_srcs), 'instrument sources:')
        else:
            # Only non-XTDF instrument sources without details enabled.
            displayed_inst_srcs = self.instrument_sources - self.detector_sources - self.legacy_sources.keys()
            print(len(displayed_inst_srcs), 'instrument sources (excluding XTDF detectors):')

        for s in sorted(displayed_inst_srcs):
            agg_str = f' [{self[s].aggregator}]' if with_aggregators else ''
            print('  -' + agg_str, s, src_alias_list(s))
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
        print(len(self.control_sources), 'control sources:')
        for s in sorted(self.control_sources):
            agg_str = f' [{self[s].aggregator}]' if with_aggregators else ''
            print('  -' + agg_str, s, src_alias_list(s))
            if any(p.match(s) for p in details_sources_re):
                # Detail for control sources: list keys
                ctrl_keys = self[s].keys(inc_timestamps=False)
                print('    - Control keys (1 entry per train):')
                keys_detail(s, sorted(ctrl_keys), prefix='      - ')

                run_keys = self._sources_data[s].files[0].get_run_keys(s)
                run_keys = {k[:-6] for k in run_keys if k.endswith('.value')}
                run_only_keys = run_keys - ctrl_keys
                if run_only_keys:
                    print('    - Additional run keys (1 entry per run):')
                    for k in sorted(run_only_keys):
                        if k in srckey_aliases[s]:
                            alias_str = ' <' + ', '.join(srckey_aliases[s][k]) + '>'
                        else:
                            alias_str = ''

                        ds = self._sources_data[s].files[0].file[
                            f"/RUN/{s}/{k.replace('.', '/')}/value"
                        ]
                        entry_shape = ds.shape[1:]
                        if entry_shape:
                            entry_info = f", entry shape {entry_shape}"
                        else:
                            entry_info = ""
                        dt = ds.dtype
                        if h5py.check_string_dtype(dt):
                            dt = 'string'
                        print(f"      - {k}{alias_str}\t[{dt}{entry_info}]")

        print()

        if self.legacy_sources:
            # Collect legacy souces matching DETECTOR_SOURCE_RE
            # separately for a condensed view.
            detector_legacy_sources = defaultdict(set)

            print(len(self.legacy_sources), 'legacy source names:')
            for s in sorted(self.legacy_sources.keys()):
                m = DETECTOR_SOURCE_RE.match(s)

                if m is not None:
                    detector_legacy_sources[m[1]].add(s)
                else:
                    # Only print non-XTDF legacy sources.
                    print(' -', s, '->', self.legacy_sources[s])

            for legacy_det, legacy_sources in detector_legacy_sources.items():
                canonical_mod = self.legacy_sources[next(iter(legacy_sources))]
                canonical_det = DETECTOR_SOURCE_RE.match(canonical_mod)[1]

                print(' -', f'{legacy_det}/*', '->', f'{canonical_det}/*',
                      f'({len(legacy_sources)})')
            print()

    def plot_missing_data(self, min_saved_pct=95, expand_instrument=False):
        """Plot sources that have missing data for some trains.

        Example output:

        .. image:: _static/plot_missing_data.png

        Parameters
        ----------

        min_saved_pct: int or float, optional
            Only show sources with less than this percentage of trains saved.
        expand_instrument: bool, optional
            Show subsections within INSTRUMENT groups. These sections usually
            have the same data missing, but it's possible for them to differ.
        """
        n_trains = len(self.train_ids)

        # Helper function that returns an alias for a source if one is
        # available, and the source name otherwise.
        def best_src_name(src):
            for alias, alias_ident in self._aliases.items():
                if isinstance(alias_ident, str) and alias_ident == src:
                    return alias

            return src

        # Check how much data is missing for each source
        run_tids = np.array(self.train_ids)
        start = time.time()
        counts = { }
        for src in self.all_sources:
            srcdata = self[src]
            if expand_instrument and srcdata.is_instrument:
                for group in srcdata.index_groups:
                    counts[f"{best_src_name(src)} {group}.*"] = \
                        srcdata.data_counts(labelled=False, index_group=group)
            elif not srcdata.is_run_only:
                counts[best_src_name(src)] = srcdata.data_counts(labelled=False)

            # Warn the user if the function will take longer than a couple seconds
            if start is not None and (time.time() - start) > 2:
                print(f"Checking sources in {len(self.files)} files, this may take a minute...")
                # Set the start time to a dummy value so the message will
                # never be printed again.
                start = None

        # Identify the sources with less than min_saved_pct% of trains
        flaky_sources = {}
        save_pcts = {}
        for name, cnt in counts.items():
            src_tids = run_tids[cnt > 0]
            save_pct = len(src_tids) / n_trains * 100

            if save_pct <= min_saved_pct:
                flaky_sources[name] = src_tids
                save_pcts[name] = save_pct

        # Sort the flaky sources by decreasing order of how many trains they're missing
        flaky_sources = dict(sorted(
            flaky_sources.items(), key=lambda x: (len(x[1]), x[0]), reverse=True
        ))

        # Plot missing data
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        fig, ax = plt.subplots(figsize=(9, max(3, len(flaky_sources) / 3.5)))

        bar_height = 0.5
        for i, src in enumerate(flaky_sources):
            # First find all the trains that are missing
            save_line = np.zeros(n_trains).astype(bool)
            save_line[np.intersect1d(self.train_ids, flaky_sources[src], return_indices=True)[1]] = True

            # Loop over each train to find blocks of trains that are either
            # present or missing.
            bars = { }
            block_start = 0
            for idx in range(n_trains):
                if save_line[idx] != save_line[block_start]:
                    # If we find a train that doesn't match the save status of
                    # the current block, create a new entry in `bars` to record
                    # the start index, the block length, and the save status.
                    bars[(block_start, idx - block_start)] = save_line[block_start]
                    block_start = idx

            # Add the last block
            bars[(block_start, n_trains - block_start)] = save_line[block_start]

            # Plot all the blocks
            ax.broken_barh(bars.keys(),
                           (i, bar_height),
                           color=["deeppink" if x else "k" for x in bars.values()])

        # Set labels and ticks
        tick_labels = [f"{src} ({save_pcts[src]:.2f}%)"
                  for i, (src, tids) in enumerate(flaky_sources.items())]
        ax.set_yticks(np.arange(len(flaky_sources)) + bar_height / 2,
                      labels=tick_labels, fontsize=8)
        ax.set_xlabel("Train ID index")

        # Set title
        title = f"Sources with less than {min_saved_pct}% of trains saved"
        run_meta = self.run_metadata()
        if "proposalNumber" in run_meta and "runNumber" in run_meta:
            title += f" in p{run_meta['proposalNumber']}, run {run_meta['runNumber']}"
        ax.set_title(title, pad=25 + len(flaky_sources) * 0.25)

        # Create legend
        legend_elements = [Line2D([0], [0], marker='o', color='w', label=label,
                                  markerfacecolor=c, markersize=6)
                           for c, label in [("k", "Missing"), ("deeppink", "Present")]]

        # bbox_factor is a variable that tries to scale down the bounding box of
        # the legend as the height of the plot grows with more sources. It's
        # necessary because the bounding box coordinates are relative to the
        # plot size, so with a tall plot the figure/legend padding will be
        # massive. 7000 is a magic number that seems to give good results.
        bbox_factor = 1 - len(flaky_sources) / 7000
        ax.legend(handles=legend_elements,
                  bbox_to_anchor=(0, 1.02 * bbox_factor, 1, 0.1 * bbox_factor),
                  loc='lower center',
                  ncol=2, borderaxespad=0)

        fig.tight_layout()

        return ax

    def detector_info(self, source):
        """Get statistics about the detector data.

        Returns a dictionary with keys:
        - 'dims' (pixel dimensions)
        - 'frames_per_train' (estimated from one file)
        - 'total_frames' (estimated assuming all trains have data)
        """
        source_files = self._sources_data[source].files
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
        files = [f for f in self.files
                 if f.has_train_ids([train_id], self.inc_suspect_trains)]
        ctrl = set().union(*[f.control_sources for f in files])
        inst = set().union(*[f.instrument_sources for f in files])

        # disp
        print('Train [{}] information'.format(train_id))
        print('Devices')
        print('\tInstruments')
        [print('\t-', d) for d in sorted(inst)] or print('\t-')
        print('\tControls')
        [print('\t-', d) for d in sorted(ctrl)] or print('\t-')

    def train_timestamps(self, labelled=False, *, pydatetime=False, euxfel_local_time=False):
        """Get approximate timestamps for each train

        Timestamps are stored and returned in UTC by default.
        Older files (before format version 1.0) do not have timestamp data,
        and the returned data in those cases will have the special value NaT
        (Not a Time).

        If *labelled* is True, they are returned in a pandas series, labelled
        with train IDs. If *pydatetime* is True, a list of Python datetime
        objects (truncated to microseconds) is returned, the same length as
        data.train_ids. Otherwise (by default), timestamps are returned as a
        NumPy array with datetime64 dtype.

        *euxfel_local_time* can be True when either *labelled* or *pydatetime* is True.
        In this case, timestamps are converted to the `Europe/Berlin` timezone.
        """
        arr = np.zeros(len(self.train_ids), dtype=np.uint64)
        id_to_ix = {tid: i for (i, tid) in enumerate(self.train_ids)}
        missing_tids = np.array(self.train_ids)
        for fa in self.files:
            tids, file_ixs, _ = np.intersect1d(
                fa.train_ids, missing_tids, return_indices=True
            )
            if not self.inc_suspect_trains:
                valid = fa.validity_flag[file_ixs]
                tids, file_ixs = tids[valid], file_ixs[valid]
            if tids.size == 0 or 'INDEX/timestamp' not in fa.file:
                continue
            file_tss = fa.file['INDEX/timestamp'][:]
            for tid, ts in zip(tids, file_tss[file_ixs]):
                arr[id_to_ix[tid]] = ts

            missing_tids = np.setdiff1d(missing_tids, tids)
            if missing_tids.size == 0:  # We've got a timestamp for every train
                break

        arr = arr.astype('datetime64[ns]')
        epoch = np.uint64(0).astype('datetime64[ns]')
        arr[arr == epoch] = 'NaT'  # Not a Time
        if labelled:
            import pandas as pd
            series = pd.Series(arr, index=self.train_ids).dt.tz_localize('UTC')
            return series.dt.tz_convert('Europe/Berlin') if euxfel_local_time else series
        elif pydatetime:
            from datetime import datetime, timezone
            res = []
            for npdt in arr:
                pydt = npdt.astype('datetime64[ms]').item()
                if pydt is not None:  # Numpy NaT becomes None
                    pydt = pydt.replace(tzinfo=timezone.utc)
                    if euxfel_local_time:
                        from zoneinfo import ZoneInfo
                        pydt = pydt.astimezone(ZoneInfo('Europe/Berlin'))
                res.append(pydt)
            return res
        elif euxfel_local_time:
            raise ValueError(
                'The euxfel_local_time option '
                + 'can only be used if either labelled or pydatetime '
                + 'are set to True'
            )
        return arr

    def run_metadata(self) -> dict:
        """Get a dictionary of metadata about the run

        From file format version 1.0, the files capture: creationDate,
        daqLibrary, dataFormatVersion, karaboFramework, proposalNumber,
        runNumber, sequenceNumber, updateDate.
        """
        if not self.is_single_run:
            raise MultiRunError()

        return self.files[0].metadata()

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
        <https://rtd.xfel.eu/docs/data-analysis-user-documentation/en/latest/datafiles.html#combining-detector-data-from-multiple-modules>`__.
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
    def __init__(
        self, data, require_all=True, flat_keys=False, keep_dims=False):
        self.data = data
        self.require_all = require_all
        self.keep_dims = keep_dims
        # {(source, key): (f, dataset)}
        self._datasets_cache = {}

        self._set_result = self._set_result_flat if flat_keys \
            else self._set_result_nested

    @staticmethod
    def _set_result_nested(res, source, key, value):
        try:
            res[source][key] = value
        except KeyError:
            res[source] = {key: value}

    @staticmethod
    def _set_result_flat(res, source, key, value):
        res[(source, key)] = value

    def _find_data(self, source, key, tid):
        file, ds = self._datasets_cache.get((source, key), (None, None))
        if ds:
            ixs = (file.train_ids == tid).nonzero()[0]
            if self.data.inc_suspect_trains and ixs.size > 0:
                return file, ixs[0], ds

            for ix in ixs:
                if file.validity_flag[ix]:
                    return file, ix, ds

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
            self._set_result(res, source, 'metadata',
                             {'source': source, 'timestamp.tid': tid})

            for key in self.data.keys_for_source(source):
                file, pos, ds = self._find_data(source, key, tid)
                if ds is None:
                    continue

                firsts, counts = file.get_index(source, '')
                first, count = firsts[pos], counts[pos]
                if not count:
                    continue

                self._set_result(res, source, key, ds[first])

        for source in self.data.instrument_sources:
            self._set_result(res, source, 'metadata',
                             {'source': source, 'timestamp.tid': tid})
            for key in self.data.keys_for_source(source):
                file, pos, ds = self._find_data(source, key, tid)
                if ds is None:
                    continue
                group = key.partition('.')[0]
                firsts, counts = file.get_index(source, group)
                first, count = firsts[pos], counts[pos]
                if count == 1 and not self.keep_dims:
                    self._set_result(res, source, key, ds[first])
                elif count > 0:
                    self._set_result(res, source, key,
                                     ds[first : first + count])

        return res

    def __iter__(self):
        for tid in self.data.train_ids:
            tid = int(tid)  # Convert numpy int to regular Python int
            if self.require_all and self.data._check_data_missing(tid):
                continue
            yield tid, self._assemble_data(tid)


def H5File(path, *, inc_suspect_trains=True):
    """Open a single HDF5 file generated at European XFEL.

    ::

        file = H5File("RAW-R0017-DA01-S00000.h5")

    Returns a :class:`DataCollection` object.

    Parameters
    ----------
    path: str
        Path to the HDF5 file
    inc_suspect_trains: bool
        If False, suspect train IDs within a file are skipped.
        In newer files, trains where INDEX/flag are 0 are suspect. For older
        files which don't have this flag, out-of-sequence train IDs are suspect.
        If True (default), it tries to include these trains.
    """
    return DataCollection.from_path(path, inc_suspect_trains=inc_suspect_trains)


def RunDirectory(
        path, include='*', file_filter=locality.lc_any, *, inc_suspect_trains=True,
        parallelize=True, _use_voview=True,
):
    """Open a European XFEL run directory.

    ::

        run = RunDirectory("/gpfs/exfel/exp/XMPL/201750/p700000/raw/r0001")

    A run directory contains a number of HDF5 files with data from the
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
    inc_suspect_trains: bool
        If False, suspect train IDs within a file are skipped.
        In newer files, trains where INDEX/flag are 0 are suspect. For older
        files which don't have this flag, out-of-sequence train IDs are suspect.
        If True (default), it tries to include these trains.
    parallelize: bool
        Enable or disable opening files in parallel. Particularly useful if
        creating child processes is not allowed (e.g. in a daemonized
        :class:`multiprocessing.Process`).
    """
    files = [f for f in os.listdir(path)
             if f.endswith('.h5') and (f.lower() != 'overview.h5')]
    files = [osp.join(path, f) for f in fnmatch.filter(files, include)]
    sel_files = file_filter(files)
    if not sel_files:
        raise FileNotFoundError(
            f"No HDF5 files found in {path} with glob pattern {include}")

    if _use_voview and (sel_files == files):
        voview_file_acc = voview.find_file_valid(path)
        if voview_file_acc is not None:
            return DataCollection([voview_file_acc],
                                  is_single_run=True,
                                  ctx_closes=True)

    files_map = RunFilesMap(path)
    t0 = time.monotonic()
    d = DataCollection.from_paths(
        sel_files, files_map, inc_suspect_trains=inc_suspect_trains,
        is_single_run=True, parallelize=parallelize
    )
    log.debug("Opened run with %d files in %.2g s",
              len(d.files), time.monotonic() - t0)
    files_map.save(d.files)

    return d

# RunDirectory was previously RunHandler; we'll leave it accessible in case
# any code was already using it.
RunHandler = RunDirectory

DEFAULT_ALIASES_FILE = "{}/usr/extra-data-aliases.yml"

def open_run(
        proposal, run, data='default', include='*', file_filter=locality.lc_any, *,
        inc_suspect_trains=True, parallelize=True, aliases=DEFAULT_ALIASES_FILE,
        _use_voview=True,
):
    """Access European XFEL data by proposal and run number.

    ::

        run = open_run(proposal=700000, run=1)

    Returns a :class:`DataCollection` object. This finds the run directory in
    standard paths on EuXFEL infrastructure.

    Parameters
    ----------
    proposal: str, int
        A proposal number, such as 2012, '2012', 'p002012', or a path such as
        '/gpfs/exfel/exp/SPB/201701/p002012'.
    run: str, int
        A run number such as 243, '243' or 'r0243'.
    data: str or Sequence of str
        'raw', 'proc' (processed), or any other location relative to the
        proposal path with data per run to access. May also be 'default'
        (combining raw & proc), 'all' (combined but preferring proc where source
        names match) or a sequence of strings to load data from
        several locations, with later locations overwriting sources present
        in earlier ones.
    include: str
        Wildcard string to filter data files.
    file_filter: callable
        Function to subset the list of filenames to open.
        Meant to be used with functions in the extra_data.locality module.
    inc_suspect_trains: bool
        If False, suspect train IDs within a file are skipped.
        In newer files, trains where INDEX/flag are 0 are suspect. For older
        files which don't have this flag, out-of-sequence train IDs are suspect.
        If True (default), it tries to include these trains.
    parallelize: bool
        Enable or disable opening files in parallel. Particularly useful if
        creating child processes is not allowed (e.g. in a daemonized
        :class:`multiprocessing.Process`).
    aliases: str, Path
        Path to an alias file for the run, see the documentation for
        :meth:`DataCollection.with_aliases` for details. If the
        argument is a string with a format argument like
        ``{}/path/to/aliases.yml``, then the format argument will be replaced with
        the proposal directory path. By default it looks for a file named
        ``{}/usr/extra-data-aliases.yml``.
    """
    absence_ok = set()
    if data == 'default':
        data = ['proc', 'raw']
        absence_ok = {'proc'}
    elif data == 'all':
        data = ['raw', 'proc']

    if isinstance(data, Sequence) and not isinstance(data, str):
        base_dc = None

        for origin in data:
            try:
                # Attempt to open data at this origin, but this may not
                # exist.
                origin_dc = open_run(
                    proposal, run, data=origin, include=include,
                    file_filter=file_filter, inc_suspect_trains=inc_suspect_trains,
                    parallelize=parallelize, aliases=aliases, _use_voview=_use_voview,
                )
            except FileNotFoundError:
                if origin not in absence_ok:
                    if base_dc is None:
                        raise
                    warn(f'No data available for this run at origin {origin}')
                continue

            if base_dc is None:  # First origin found
                base_dc = origin_dc
                continue

            # Deselect to those sources in the base not present in
            # this origin.
            base_extra = base_dc.deselect(
                [(src, '*') for src
                in base_dc.all_sources & origin_dc.all_sources])

            if base_extra.files:
                # If base is not a subset of this origin, merge the
                # "extra" base sources into the origin sources and
                # re-enable is_single_run flag.
                base_dc = origin_dc.union(base_extra)
                base_dc.is_single_run = True
            else:
                # If the sources we previously found are a subset of those
                # in the latest origin, discard the previous data.
                base_dc = origin_dc

        return base_dc

    if isinstance(proposal, os.PathLike):
        prop_dir = os.fsdecode(proposal)
    else:
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

    dc = RunDirectory(
        osp.join(prop_dir, data, run), include=include, file_filter=file_filter,
        inc_suspect_trains=inc_suspect_trains, parallelize=parallelize,
        _use_voview=_use_voview,
    )

    # Normalize string arguments to be an absolute Path
    if isinstance(aliases, str):
        aliases = Path(aliases.format(prop_dir))

    # If we're using the default aliases file and it doesn't exist, ignore it
    # without throwing any errors.
    default_aliases = Path(DEFAULT_ALIASES_FILE.format(prop_dir))
    if aliases == default_aliases and not default_aliases.is_file():
        aliases = None

    if aliases is not None:
        dc = dc.with_aliases(aliases)
        log.info("Loading %d aliases from: %s", len(dc._aliases), aliases)

    return dc
