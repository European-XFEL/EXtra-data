"""Internal module for accessing EuXFEL HDF5 files

This includes convenience features for getting the metadata & indexes from a
file, as well as machinery to close less recently accessed files, so we don't
run into the limit on the number of open files.
"""
from collections import defaultdict, OrderedDict
import h5py, h5py.h5o
import numpy as np
import os
import os.path as osp
import re
import resource
from warnings import warn
from weakref import WeakValueDictionary

from .exceptions import FileStructureError, SourceNameError

# Track all FileAccess objects - {path: FileAccess}
file_access_registry = WeakValueDictionary()


class OpenFilesLimiter(object):
    """
    Working with FileAccess, keep the number of opened HDF5 files
    under the given limit by closing files accessed longest time ago.
    """
    def __init__(self, maxfiles=128):
        self._maxfiles = maxfiles
        # We don't use the values, but OrderedDict is a handy as a queue
        # with efficient removal of entries by key.
        self._cache = OrderedDict()

    @property
    def maxfiles(self):
        return self._maxfiles

    @maxfiles.setter
    def maxfiles(self, maxfiles):
        """Set the new file limit and closes files over the limit"""
        self._maxfiles = maxfiles
        self.close_old_files()

    def _check_files(self):
        # Discard entries from self._cache if their FileAccess no longer exists
        self._cache = OrderedDict.fromkeys(
            path for path in self._cache if path in file_access_registry
        )

    def n_open_files(self):
        self._check_files()
        return len(self._cache)

    def close_old_files(self):
        if len(self._cache) <= self.maxfiles:
            return

        # Now check how many paths still have an existing FileAccess object
        n = self.n_open_files()
        while n > self.maxfiles:
            path, _ = self._cache.popitem(last=False)
            file_access = file_access_registry.get(path, None)
            if file_access is not None:
                file_access.close()
            n -= 1

    def touch(self, filename):
        """
        Add/move the touched file to the end of the `cache`.

        If adding a new file takes it over the limit of open files, another file
        will be closed.

        For use of the file cache, FileAccess should use `touch(filename)` every time
        it provides the underlying instance of `h5py.File` for reading.
        """
        if filename in self._cache:
            self._cache.move_to_end(filename)
        else:
            self._cache[filename] = None
            self.close_old_files()

    def closed(self, filename):
        """Discard a closed file from the cache"""
        self._cache.pop(filename, None)


def init_open_files_limiter():
    # Raise the limit for open files (1024 -> 4096 on Maxwell)
    nofile = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (nofile[1], nofile[1]))
    maxfiles = nofile[1] // 2
    return OpenFilesLimiter(maxfiles)

open_files_limiter = init_open_files_limiter()


class MetaFileAccess(type):
    # Override regular instance creation to check in the registry first.
    # Defining __new__ on the class is not enough, because an object retrieved
    # from the registry that way will have its __init__ run again.
    def __call__(cls, filename, _cache_info=None):
        filename = osp.abspath(filename)
        inst = file_access_registry.get(filename, None)
        if inst is None:
            inst = file_access_registry[filename] = type.__call__(
                cls, filename, _cache_info
            )

        return inst


class FileAccess(metaclass=MetaFileAccess):
    """Access an EuXFEL HDF5 file.

    This does not necessarily keep the real file open, but opens it on demand.
    It assumes that the file is not changing on disk while this object exists.

    Parameters
    ----------
    filename: str
        A path to an HDF5 file
    """
    _file = None
    _format_version = None
    _path_infos = None
    _filename_infos = None
    metadata_fstat = None

    # Regular expressions to extract path and filename information for HDF5
    # files saved on the EuXFEL computing infrastructure.
    euxfel_path_pattern = re.compile(
        # A path may have three different prefixes depending on the storage location.
        r'\/(gpfs\/exfel\/exp|gpfs\/exfel\/d|pnfs\/xfel.eu\/exfel\/archive\/XFEL)'

        # The first prefix above uses the pattern <instrument>/<cycle>/<proposal>/<class>/<run>,
        # the other two use <class>/<instrument>/<cycle>/<proposal>
        r'\/(\w+)\/(\d{6}|\w+)\/(\d{6}|p\d{6})\/(p\d{6}|[a-z]+)\/r\d{4}')

    euxfel_filename_pattern = re.compile(r'([A-Z]+)-R\d{4}-(\w+)-S(\d{5}).h5')

    def __init__(self, filename, _cache_info=None):
        self.filename = osp.abspath(filename)

        if _cache_info:
            self.train_ids = _cache_info['train_ids']
            self.control_sources = _cache_info['control_sources']
            self.instrument_sources = _cache_info['instrument_sources']
            self.legacy_sources = _cache_info['legacy_sources']
            self.validity_flag = _cache_info['flag']
        else:
            try:
                tid_data = self.file['INDEX/trainId'][:]
            except KeyError:
                raise FileStructureError('INDEX/trainId dataset not found')

            self.train_ids = tid_data[tid_data != 0]

            self.control_sources, self.instrument_sources, self.legacy_sources \
                = self._read_data_sources()

            self.validity_flag = None

        if self.validity_flag is None:
            if self.format_version == '0.5':
                self.validity_flag = self._guess_valid_trains()
            else:
                self.validity_flag = self.file['INDEX/flag'][:len(self.train_ids)].astype(bool)

                if self.format_version == '1.1':
                    # File format version 1.1 changed the semantics of
                    # INDEX/flag from a boolean flag to an index, with
                    # the time server device being hardcoded to occur
                    # at index 0. Inverting the flag after the boolean
                    # cast above restores compatibility with format
                    # version 1.0, with any "invalid" train having an
                    # index >= 1, thus being casted to True and inverted
                    # to False. Format 1.2 restored the 1.0 semantics.
                    np.logical_not(self.validity_flag, out=self.validity_flag)

                    warn(
                        'Train validation is not fully supported for data '
                        'format version 1.1. If you have issues accessing '
                        'these files, please contact da-support@xfel.eu.',
                        stacklevel=2
                    )

        if self._file is not None:
            # Store the stat of the file as it was when we read the metadata.
            # This is used by the run files map.
            self.metadata_fstat = os.stat(self.file.id.get_vfd_handle())

        # {(file, source, group): (firsts, counts)}
        self._index_cache = {}
        # {source: set(keys)}
        self._keys_cache = {}
        self._run_keys_cache = {}
        # {source: set(keys)} - including incomplete sets
        self._known_keys = defaultdict(set)

    @property
    def file(self):
        open_files_limiter.touch(self.filename)
        # Local var to avoid a race condition when another thread calls .close()
        file = self._file
        if file is None:
            file = self._file = h5py.File(self.filename, 'r')

        return file

    def _evaluate_path_infos(self):
        m = self.euxfel_path_pattern.match(osp.dirname(osp.realpath(self.filename)))

        if m:
            if m[1] == 'gpfs/exfel/exp':
                self._path_infos = (m[5], m[2], m[3])  # ONC path.
            else:
                self._path_infos = (m[2], m[3], m[4])  # Maxwell path.
        else:
            self._path_infos = (None, None, None)

    def _evaluate_filename_infos(self):
        m = self.euxfel_filename_pattern.match(osp.basename(self.filename))

        if m:
            self._filename_infos = (m[1], m[2], int(m[3]))
        else:
            self._filename_infos = (None, None, None)

    @property
    def storage_class(self):
        if self._path_infos is None:
            self._evaluate_path_infos()
        return self._path_infos[0]

    @property
    def instrument(self):
        if self._path_infos is None:
            self._evaluate_path_infos()
        return self._path_infos[1]

    @property
    def cycle(self):
        if self._path_infos is None:
            self._evaluate_path_infos()
        return self._path_infos[2]

    @property
    def data_category(self):
        if self._filename_infos is None:
            self._evaluate_filename_infos()
        return self._filename_infos[0]

    @property
    def aggregator(self):
        if self._filename_infos is None:
            self._evaluate_filename_infos()
        return self._filename_infos[1]

    @property
    def sequence(self):
        if self._filename_infos is None:
            self._evaluate_filename_infos()
        return self._filename_infos[2]

    @property
    def valid_train_ids(self):
        return self.train_ids[self.validity_flag]

    def has_train_ids(self, tids: np.ndarray, inc_suspect=False):
        f_tids = self.train_ids if inc_suspect else self.valid_train_ids
        return np.intersect1d(tids, f_tids).size > 0

    def close(self):
        """Close* the HDF5 file this refers to.

        The file may not actually be closed if there are still references to
        objects from it, e.g. while iterating over trains. This is what HDF5
        calls 'weak' closing.
        """
        if self._file:
            self._file = None
        open_files_limiter.closed(self.filename)

    @property
    def format_version(self):
        if self._format_version is None:
            version_ds = self.file.get('METADATA/dataFormatVersion')
            if version_ds is not None:
                self._format_version = version_ds[0].decode('ascii')
            else:
                # The first version of the file format had no version number.
                # Numbering started at 1.0, so we call the first version 0.5.
                self._format_version = '0.5'

        return self._format_version

    def _read_data_sources(self):
        control_sources, instrument_sources, legacy_sources = set(), set(), dict()

        # The list of data sources moved in file format 1.0
        if self.format_version == '0.5':
            data_sources_path = 'METADATA/dataSourceId'
        else:
            data_sources_path = 'METADATA/dataSources/dataSourceId'

        try:
            data_sources_group = self.file[data_sources_path]
        except KeyError:
            raise FileStructureError(f'{data_sources_path} not found')

        for source_id in data_sources_group[:]:
            if not source_id:
                continue
            source_id = source_id.decode()
            category, _, h5_source = source_id.partition('/')
            if category == 'INSTRUMENT':
                device, _, chan_grp = h5_source.partition(':')
                chan, _, group = chan_grp.partition('/')
                source = device + ':' + chan

                if source not in instrument_sources:
                    # The legacy source name is only expected to be used
                    # for instrument (more precisely, XTDF sources) for
                    # now. For performance reasons, the check is
                    # therefore only performed here, and only once rather
                    # than by index group.
                    item = self.file.get(f'{category}/{source}', getlink=True)

                    if isinstance(item, h5py.SoftLink):
                        legacy_sources[source] = item.path[1:].partition('/')[2]

                    instrument_sources.add(source)
                # TODO: Do something with groups?
            elif category == 'CONTROL':
                control_sources.add(h5_source)
            elif category == 'Karabo_TimerServer':
                # Ignore virtual data source used only in file format
                # version 1.1 / pclayer-1.10.3-2.10.5.
                pass
            else:
                raise ValueError("Unknown data category %r" % category)

        return frozenset(control_sources), frozenset(instrument_sources), \
            legacy_sources

    def _guess_valid_trains(self):
        # File format version 1.0 includes a flag which is 0 if a train ID
        # didn't come from the time server. We use this to skip bad trains,
        # especially for AGIPD.
        # Older files don't have this flag, so this tries to estimate validity.
        # The goal is to have a monotonic sequence within the file with the
        # fewest trains skipped.
        train_ids = self.train_ids
        flag = np.ones_like(train_ids, dtype=bool)

        for ix in np.nonzero(train_ids[1:] <= train_ids[:-1])[0]:
            # train_ids[ix] >= train_ids[ix + 1]
            invalid_before = train_ids[:ix+1] >= train_ids[ix+1]
            invalid_after = train_ids[ix+1:] <= train_ids[ix]
            # Which side of the downward jump in train IDs would need fewer
            # train IDs invalidated?
            if np.count_nonzero(invalid_before) < np.count_nonzero(invalid_after):
                flag[:ix+1] &= ~invalid_before
            else:
                flag[ix+1:] &= ~invalid_after

        return flag

    def __hash__(self):
        return hash(self.filename)

    def __eq__(self, other):
        return isinstance(other, FileAccess) and (other.filename == self.filename)

    def __repr__(self):
        return "{}({})".format(type(self).__name__, repr(self.filename))

    # This is a staticmethod rather than a classmethod, because cloudpickle
    # (used in Dask, clusterfutures, etc.) tries to save bound classmethods
    # 'by value', which includes the registry, and that goes wrong.
    # The staticmethod is pickled 'by reference', avoiding that issue.
    @staticmethod
    def _from_pickle(filename, state):
        """Called when an instance is loaded from a pickle"""
        inst = file_access_registry.get(filename, None)
        if inst is None:
            inst = file_access_registry[filename] = FileAccess.__new__(FileAccess)
            inst.__dict__.update(state)

        return inst

    def __reduce__(self):
        """Called when a `FileAccess` instance is pickled"""
        state = self.__dict__.copy()
        state['_file'] = None
        return self._from_pickle, (self.filename, state)

    @property
    def all_sources(self):
        return self.control_sources | self.instrument_sources

    def get_index(self, source, group):
        """Get first index & count for a source and for a specific train ID.

        Indices are cached; this appears to provide some performance benefit.
        """
        try:
            return self._index_cache[(source, group)]
        except KeyError:
            ix = self._read_index(source, group)
            self._index_cache[(source, group)] = ix
            return ix

    def _read_index(self, source, group):
        """Get first index & count for a source.

        This is 'real' reading when the requested index is not in the cache.
        """
        ntrains = len(self.train_ids)
        ix_group = self.file['/INDEX/{}/{}'.format(source, group)]
        firsts = ix_group['first'][:ntrains]
        if 'count' in ix_group:
            counts = ix_group['count'][:ntrains]
        else:
            status = ix_group['status'][:ntrains]
            counts = np.uint64((ix_group['last'][:ntrains] - firsts + 1) * status)
        return firsts, counts

    def index_groups(self, source):
        """Get group names for index data, to use with .get_index()"""
        if source in self.control_sources:
            return {''}
        elif source in self.instrument_sources:
            return set(self.file[f'/INDEX/{source}'].keys())
        else:
            raise SourceNameError(source)

    def metadata(self) -> dict:
        """Get the contents of the METADATA group as a dict

        Not including the lists of data sources
        """
        if self.format_version == '0.5':
            # Pretend this is actually there, like format version 1.0
            return {'dataFormatVersion': '0.5'}

        r = {}
        for k, ds in self.file['METADATA'].items():
            if not isinstance(ds, h5py.Dataset):
                continue
            v = ds[0]
            if isinstance(v, bytes):
                v = v.decode('utf-8', 'surrogateescape')
            r[k] = v
        return r

    def get_keys(self, source):
        """Get keys for a given source name

        Keys are found by walking the HDF5 file, and cached for reuse.
        """
        try:
            return self._keys_cache[source]
        except KeyError:
            pass

        if source in self.control_sources:
            group = '/CONTROL/' + source
        elif source in self.instrument_sources:
            group = '/INSTRUMENT/' + source
        else:
            raise SourceNameError(source)

        res = set()

        def add_key(key, value):
            if isinstance(value, h5py.Dataset):
                res.add(key.replace('/', '.'))

        self.file[group].visititems(add_key)
        self._keys_cache[source] = res
        return res

    def get_one_key(self, source, index_group=None):
        """Similar to get_keys(), except it returns only a single key for performance"""

        # Use empty prefix if no index group filter is active.
        prefix = index_group + '.' if index_group else ''

        if source in self._keys_cache:
            for key in self._keys_cache[source]:
                if key.startswith(prefix):
                    return key

            # _keys_cache is a complete set, so this point can only be
            # reached for a key-less source or a non-existing index
            # group was passed.
            if self._keys_cache[source]:
                raise ValueError(f'{index_group} not an index group of '
                                 f'`{source}`')

        if self._known_keys[source]:
            for key in self._known_keys[source]:
                if key.startswith(prefix):
                    return key

        if source in self.control_sources:
            root = 'CONTROL'
        elif source in self.instrument_sources:
            root = 'INSTRUMENT'
        else:
            raise SourceNameError(source)

        group = f'/{root}/{source}'

        if index_group:
            group += '/' + index_group

        def get_key(subkey, value):
            if isinstance(value, h5py.Dataset):
                return prefix + subkey.replace('/', '.')

        try:
            h5_group = self.file[group]
        except KeyError:
            # Can only happen for missing index groups, as missing
            # sources are handled above already.
            raise ValueError(f'{index_group} not an index group of `{source}`')
        else:
            return h5_group.visititems(get_key)

    def get_run_keys(self, source):
        """Get the keys in the RUN section for a given control source name

        Keys are found by walking the HDF5 file, and cached for reuse.
        """
        try:
            return self._run_keys_cache[source]
        except KeyError:
            pass

        if source not in self.control_sources:
            raise SourceNameError(source)

        res = set()

        def add_key(key, value):
            if isinstance(value, h5py.Dataset):
                res.add(key.replace('/', '.'))

        self.file['/RUN/' + source].visititems(add_key)
        self._run_keys_cache[source] = res
        return res

    def has_source_key(self, source, key):
        """Check if the given source and key exist in this file

        This doesn't scan for all the keys in the source, as .get_keys() does.
        """
        try:
            return key in self._keys_cache[source]
        except KeyError:
            pass

        if key in self._known_keys[source]:
            return True

        if source in self.control_sources:
            path = '/CONTROL/{}/{}'.format(source, key.replace('.', '/'))
        elif source in self.instrument_sources:
            path = '/INSTRUMENT/{}/{}'.format(source, key.replace('.', '/'))
        else:
            raise SourceNameError(source)

        # self.file.get(path, getclass=True) works, but is weirdly slow.
        # Checking like this is much faster.
        if (path in self.file) and isinstance(
                h5py.h5o.open(self.file.id, path.encode()), h5py.h5d.DatasetID
        ):
            self._known_keys[source].add(key)
            return True
        return False

    def dset_proxy(self, ds_path: str):
        return DatasetProxy(self, ds_path)


class DatasetProxy:
    """A picklable reference to an HDF5 dataset, suitable for dask.array

    Dask tries to do this automatically for h5py Dataset objects, but with
    some limitations:

    - It only works with Dask distributed, not Dask's local schedulers.
    - Dask storing references to h5py Datasets keeps the files open, breaking
      our attempts to manage the number of open files.
    """
    def __init__(self, file_acc: FileAccess, ds_path: str):
        # We could just store the file name and use h5py on demand, but storing
        # our FileAccess object lets it use our cache of open files.
        self.file_acc = file_acc
        self.ds_path = ds_path
        ds = file_acc.file[ds_path]

        # dask.array expects these three array-like attributes:
        self.shape = ds.shape
        self.ndim = ds.ndim
        self.dtype = ds.dtype

    def __getitem__(self, item):
        return self.file_acc.file[self.ds_path][item]
