import h5py
import numpy as np

# compatibility to future numpy features
from .numpy_future import add_future_function_into
add_future_function_into(np)


def accumulate(iterable, *, initial=None):
    """Return running totals"""
    # accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # accumulate([1,2,3,4,5], initial=100) --> 100 101 103 106 110 115
    it = iter(iterable)
    total = initial
    if initial is None:
        try:
            total = next(it)
        except StopIteration:
            return
    yield total
    for element in it:
        total += element
        yield total


# Attention! `Dataset` is the descriptor class and its instances are
# intended to be used as class members of `FileWriter` children. Changing
# them leads to changes in the host class itself and in all its instances.
# Therefore, one can change `self` only in the `__init__` method and
# in methods that are called from the `FileWriterMeta` metaclass.
class DatasetBase:
    """Base dataset descriptor class"""

    class Attributes:
        """Dataset attributes structure"""
        def __init__(self, **kwargs):
            for kw, val in kwargs.items():
                setattr(self, kw, val)

    def __init__(self, source_name, key, entry_shape, dtype,
                 chunks=None, compression=None):
        self.entry_shape = entry_shape
        self.dtype = dtype
        self.compression = compression
        self.chunks = chunks
        if source_name and source_name[0] == '@':
            self.orig_name = (True, source_name[1:], key)
        else:
            self.orig_name = (False, source_name, key)

    def set_name(self, source_name, key):
        """Sets new name to the dataset"""
        self.orig_name = (False, source_name, key)
        self.canonical_name = (source_name, key)

    def resolve_name(self, aliases={}):
        """Normalizes source name and key"""
        isalias, source_id, key = self.orig_name

        # resolve reference
        source_name = aliases[source_id] if isalias else source_id
        self.canonical_name = (source_name, key)

    def get_dataset_fullname(self, writer):
        # expected to return (source_name, key, stype)
        raise NotImplementedError

    def get_entry_attr(self, writer):
        # expected to return (entry_shape, dtype)
        return self.entry_shape, self.dtype

    def get_chunks(self, writer):
        # expected to return chunks
        return self.chunks

    def init_dataset_attr(self, writer):
        # expand names
        source_name, key, stype = self.get_dataset_fullname(writer)

        # expand entry attributes
        entry_shape, dtype = self.get_entry_attr(writer)

        # auto chunking
        chunks = self.get_chunks(writer)
        if chunks is None:
            chunks = Dataset._chunks_autosize(
                writer._meta.max_train_per_file, entry_shape,
                dtype, stype)

        writer._ds_attrs[id(self)] = Dataset.Attributes(
            source_name=source_name, key=key, stype=stype,
            entry_shape=entry_shape, dtype=dtype, chunks=chunks,
            compression=self.compression,
        )

    def check_value(self, writer, value):
        """Checks data"""
        # can we need to checked type cast?

        # shape check
        entry_shape = self(writer).entry_shape
        value_shape = np.shape(value)
        shape = np.broadcast_shapes(value_shape, (1,) + entry_shape)
        if shape == entry_shape:
            nrec = 1
        elif shape[1:] == entry_shape:
            nrec = shape[0]
        else:
            raise ValueError(f"shape mismatch: {value_shape} cannot "
                             f"be broadcast to {(None, ) + entry_shape}")
        return nrec

    @staticmethod
    def _chunks_autosize(max_trains, entry_shape, dtype, stype):
        """Caclulates chunk size"""
        MN = (max_trains, 32, 32)
        SZ = (1 << 14, 1 << 19, 1 << 23)  # 16K, 512K, 8M

        size = np.prod(entry_shape, dtype=int)
        ndim = len(entry_shape)
        nbytes = size * np.dtype(dtype).itemsize

        entry_type = int(size != 1) * (1 + int(ndim > 1))
        chunk = max(SZ[entry_type] // nbytes, MN[entry_type])
        if stype == 0:
            chunk = min(chunk, max_trains)

        return (chunk,) + tuple(entry_shape)

    def __call__(self, writer):
        return writer._ds_attrs[id(self)]

    def create(self, writer, grp):
        """Creates dataset in h5-file and return writer"""
        attr = self(writer)

        ds = grp.create_dataset(
            attr.key, (0,) + attr.entry_shape,
            dtype=attr.dtype, chunks=attr.chunks, maxshape=(None,)
            + attr.entry_shape, compression=attr.compression
        )
        if writer._meta.buffering:
            wrt = DatasetBufferedWriter(self, ds, attr.chunks)
        else:
            wrt = DatasetDirectWriter(self, ds, attr.chunks)

        return wrt


class Dataset(DatasetBase):
    """Dataset descriptor"""

    def get_attribute_setter(self, name):
        """Returns suitable attribute setter instance"""
        return DataSetter(name)

    def get_dataset_fullname(self, writer):
        # expected to return (source_name, key, stype)
        source_name, key = self.canonical_name
        return Dataset._normalize_name(
            source_name.format(**writer.param), key)

    def get_chunks(self, writer):
        # expected to return chunks
        return Dataset._expand_shape(self.chunks, writer)

    def get_entry_attr(self, writer):
        # expected to return (entry_shape, dtype)
        return (Dataset._expand_shape(self.entry_shape, writer),
                self.dtype)

    @staticmethod
    def _normalize_name(source_name, key):
        """Transforms canonical name to the internal form"""
        # can we really distinguish sources by colons?
        stype = int(':' in source_name)
        if stype:
            tk, key = key.split('.', 1)
            source_name = source_name + '/' + tk

        return source_name, key.replace('.', '/'), stype

    @staticmethod
    def _expand_shape(shape_decl, writer):
        if isinstance(shape_decl, str):
            shape = tuple(writer.param[shape_decl])
        elif hasattr(shape_decl, '__iter__'):
            shape = tuple(writer.param[n] if isinstance(n, str) else n
                          for n in shape_decl)
        else:
            shape = shape_decl

        return shape


class DatasetWriterBase:
    """Abstract class for writers wrapping h5py"""
    def __init__(self, ds, file_ds, chunks):
        self.ds = ds
        self.file_ds = file_ds
        self.chunks = chunks
        self.pos = 0

    def flush(self):
        pass

    def write(self, data, nrec, start=None):
        raise NotImplementedError


class DatasetDirectWriter(DatasetWriterBase):
    """Class writes data directly to the file"""
    def write(self, data, nrec, start=None):
        """Writes data to disk"""
        end = self.pos + nrec
        if start is None:
            start = 0
        self.file_ds.resize(end, 0)
        if not np.ndim(data):
            self.file_ds[self.pos:end] = data
        else:
            # self.file_ds[self.pos:end] = data[start:start+nrec]
            self.file_ds.write_direct(
                data, np.s_[start:start+nrec], np.s_[self.pos:end])


class DatasetBufferedWriter(DatasetWriterBase):
    """Class implements buffered writing"""
    def __init__(self, ds, file_ds, chunks):
        super().__init__(ds, file_ds, chunks)
        self._data = np.empty(chunks, dtype=ds.dtype)
        self.size = chunks[0]
        self.nbuf = 0

    def flush(self):
        """Writes buffer to disk"""
        if self.nbuf:
            end = self.pos + self.nbuf
            self.file_ds.resize(end, 0)
            self.file_ds.write_direct(
                self._data, np.s_[:self.nbuf], np.s_[self.pos:end])
            self.pos = end
            self.nbuf = 0

    def write_one(self, value):
        """Buffers single record"""
        self._data[self.nbuf] = value
        self.nbuf += 1
        if self.nbuf >= self.size:
            self.flush()

    def write_many(self, arr, nrec, start=None):
        """Buffers multiple records"""
        if start is None:
            start = 0
        buf_nrest = self.size - self.nbuf
        data_nrest = nrec - buf_nrest
        if data_nrest < 0:
            # copy
            end = self.nbuf + nrec
            self._data[self.nbuf:end] = arr[start:start+nrec]
            self.nbuf = end
        elif self.nbuf and data_nrest < self.size:
            # copy, flush, copy
            split = start + buf_nrest
            self._data[self.nbuf:] = arr[start:split]

            end = self.pos + self.size
            self.file_ds.resize(end, 0)
            self.file_ds.write_direct(
                self._data, np.s_[:], np.s_[self.pos:end])
            self.pos = end

            self._data[:data_nrest] = arr[split:start+nrec]
            self.nbuf = data_nrest
        else:
            # flush, write, copy
            nrest = nrec % self.size
            nwrite = nrec - nrest

            split = self.pos + self.nbuf
            end = split + nwrite
            self.file_ds.resize(end, 0)
            if self.nbuf:
                self.file_ds.write_direct(
                    self._data, np.s_[:self.nbuf], np.s_[self.pos:split])
            self.file_ds.write_direct(arr, np.s_[start:start+nwrite],
                                      np.s_[split:end])

            self._data[:nrest] = arr[start+nwrite:start+nrec]
            self.pos = end
            self.nbuf = nrest

    def write(self, data, nrec, start=None):
        """Buffer or writes data"""
        if nrec == 1:
            self.write_one(data[start] if start is not None else data)
        else:
            if not isinstance(data, np.ndarray):
                arr = np.array(data, dtype=self.ds.dtype)
            else:
                arr = data

            # arr = np.broadcast_to(data, (nrec,) + self.ds.entry_shape)
            self.write_many(arr, nrec, start=start)


class DataQueue:
    def __init__(self):
        self.data = []
        self.size = []
        self.nwritten = 0
        self.nready = 0

    def __bool__(self):
        return bool(self.data)

    def append(self, count, data):
        """Appends data into queue for future writing"""
        ntrain = len(count)
        self.data.append((count, data))
        self.size.append((0, 0, ntrain))
        self.nready += ntrain

    def reset(self):
        """Reset counters for new file"""
        self.nready -= self.nwritten
        self.nwritten = 0

    def get_max_below(self, end):
        """Finds the maximum number of trains that data items in the queue
        fill without splitting and below the given number"""
        ntrain = self.nwritten
        for _, _, n in self.size:
            if ntrain + n > end:
                return ntrain
            ntrain += n
        return ntrain

    def write(self, writer, end):
        """Writes data items from the queue"""
        nrest = end - self.nwritten
        while nrest > 0:
            offset, train0, ntrain = self.size[0]
            if ntrain == 1 and offset == 0:
                count, data = self.data.pop(0)
                writer.write(data, count[0])
                nrest -= 1
                self.size.pop(0)
            elif ntrain <= nrest:
                count, data = self.data.pop(0)
                trainN = train0 + ntrain
                nrec = np.sum(count[train0:trainN])
                writer.write(data, nrec, offset)

                nrest -= ntrain
                self.size.pop(0)
            else:
                count, data = self.data[0]
                trainN = train0 + nrest
                nrec = np.sum(count[train0:trainN])
                writer.write(data, nrec, offset)

                self.size[0] = (offset + nrec, trainN, ntrain - nrest)
                nrest = 0

        self.nwritten = end


# Attention! Do not instanciate `Source` in the metaclass `FileWriterMeta`
class Source:
    """Creates data source group and its indexes"""

    SECTION = ('CONTROL', 'INSTRUMENT')

    def __init__(self, writer, name, stype=None):
        self.writer = writer
        self.name = name
        if stype is None:
            self.stype = int(':' in name)
        else:
            self.stype = stype

        self.section = self.SECTION[self.stype]
        if writer._meta.break_into_sequence:
            self.max_trains = writer._meta.max_train_per_file
        else:
            self.max_trains = None

        self.ndatasets = 0
        self.nready = 0

        self.datasets = []
        self.dsno = {}
        self.file_ds = []
        self.data = []

        self.count_buf = []

        self.first = []
        self.count = []
        self.nrec = 0

        self.block_writing = True

    def add(self, name, ds):
        """Adds dataset to the source"""
        self.dsno[name] = len(self.datasets)
        self.datasets.append(ds)
        self.data.append(DataQueue())
        self.file_ds.append(None)
        self.ndatasets += 1

    def create(self):
        """Creates all datasets in file"""
        grp = self.writer._file.create_group(self.section + '/' + self.name)
        for dsno, ds in enumerate(self.datasets):
            self.file_ds[dsno] = ds.create(self.writer, grp)
        self._grp = grp
        self.block_writing = False

        while self.nready >= self.ndatasets and not self.block_writing:
            self.write_data()

        return grp

    def create_index(self, index_grp, max_trains):
        """Create source index in h5-file"""
        grp = index_grp.create_group(self.name)
        for key in ('first', 'count'):
            ds = grp.create_dataset(
                key, (max_trains,), dtype=np.uint64, chunks=(max_trains,),
                maxshape=(None,)
            )
            ds[:] = 0

    def write_index(self, index_grp, ntrains):
        """Writes source index in h5-file"""
        nmissed = ntrains - len(self.count)
        if nmissed > 0:
            self.count += [0] * nmissed
            self.first += [self.nrec] * nmissed

        grp = index_grp[self.name]
        for dsname in ('first', 'count'):
            ds = grp[dsname]
            ds.resize(ntrains, axis=0)
            val = getattr(self, dsname)
            ds[:] = val[:ntrains]
            del val[:ntrains]

        if len(self.count):
            self.first = list(accumulate(self.count[:-1], initial=0))
            self.nrec = sum(self.count)
        else:
            self.first = []
            self.nrec = 0

        del self.count_buf[:ntrains]

    def close_datasets(self):
        """Finalize writing"""
        for dsno, ds in enumerate(self.datasets):
            self.file_ds[dsno].flush()
            self.data[dsno].reset()

    def add_train_data(self, nrec, name, value):
        """Adds single train data to the source"""
        if self.stype == 0 and nrec != 1:
            raise ValueError("maximum one entry per train can be written "
                             "in control source")

        dsno = self.dsno[name]

        ntrain = self.data[dsno].nready
        nwritten = len(self.count)
        if nwritten > ntrain:
            if self.count_buf[ntrain] != nrec:
                raise ValueError("count mismatch the number of frames "
                                 "in source")
        else:
            self.count_buf.append(nrec)

        self._put_data(dsno, [nrec], value)

    def add_data(self, count, name, value):
        """Adds multitrain data to the source"""
        if self.stype == 0 and np.any(count > 1):
            raise ValueError("maximum one entry per train can be written "
                             "in control source")

        dsno = self.dsno[name]

        ntrain = self.data[dsno].nready
        nwritten = len(self.count)
        if nwritten > ntrain:
            nmatch = min(nwritten - ntrain, len(count))
            if np.any(self.count_buf[ntrain:ntrain+nmatch] != count[:nmatch]):
                raise ValueError("count mismatch the number of frames "
                                 "in source")

            self.count_buf += count[nmatch:].tolist()
        else:
            self.count_buf += count.tolist()

        self._put_data(dsno, count, value)

    def _put_data(self, dsno, count, value):
        self.nready += not self.data[dsno]
        self.data[dsno].append(count, value)

        while self.nready >= self.ndatasets and not self.block_writing:
            self.write_data()

    def write_data(self):
        """Write data when the trains completely filled"""
        train0 = len(self.count)
        max_ready = min(d.nready for d in self.data)

        if self.max_trains is not None and self.max_trains < max_ready:
            max_ready = self.max_trains
            self.block_writing = True

        self.nready = 0
        trainN = train0
        for dsno in range(self.ndatasets):
            if self.block_writing:
                end = max_ready
            else:
                end = self.data[dsno].get_max_below(max_ready)

            self.data[dsno].write(self.file_ds[dsno], end)
            self.nready += bool(self.data[dsno])
            trainN = max(trainN, end)

        count = self.count_buf[train0:trainN]
        first = list(accumulate(count[:-1], initial=self.nrec))
        self.count += count
        self.first += first
        self.nrec += np.sum(count, dtype=int)

    def get_ntrain(self, dsname):
        dsno = self.dsno[dsname]
        return self.data[dsno].nready

    def get_min_trains(self):
        return min(d.nready for d in self.data)


class Options:
    """Provides a set of options with overriding default values
    by ones declared in Meta subclass
    """
    NAMES = (
        'max_train_per_file', 'break_into_sequence',
        'class_attrs_interface', 'buffering', 'aliases'
    )

    def __init__(self, meta=None, base=None):
        self.max_train_per_file = 500
        self.break_into_sequence = False
        self.warn_on_missing_data = False
        self.class_attrs_interface = True
        self.buffering = True
        self.aliases = {}

        self.copy(base)
        self.override_defaults(meta)

    def copy(self, opts):
        if not opts:
            return
        for attr_name in Options.NAMES:
            setattr(self, attr_name, getattr(opts, attr_name))

    def override_defaults(self, meta):
        if not meta:
            return
        meta_attrs = meta.__dict__.copy()
        for attr_name in meta.__dict__:
            if attr_name.startswith('_'):
                del meta_attrs[attr_name]

        for attr_name in Options.NAMES:
            if attr_name in meta_attrs:
                val = meta_attrs.pop(attr_name)
                if isinstance(val, dict):
                    getattr(self, attr_name).update(val)
                else:
                    setattr(self, attr_name, val)

        if meta_attrs != {}:
            raise TypeError("'class Meta' got invalid attribute(s): " +
                            ','.join(meta_attrs))


class MultiTrainData:
    def __init__(self, count, data):
        self.count = count
        self.data = data


class DataSetterBase:
    def __set__(self, instance, value):
        raise NotImplementedError


class BlockedSetter(DataSetterBase):
    def __set__(self, instance, value):
        raise RuntimeError(
            "Class attributes interface is disabled. Use option "
            "'class_attrs_interface=True' to enable it.")


class DataSetter(DataSetterBase):
    """Overrides the setters for attributes which declared as datasets
    in order to use the assignment operation for adding data in a train
    """
    def __init__(self, name):
        self.name = name

    def __set__(self, instance, value):
        if isinstance(value, MultiTrainData):
            instance.add_value(value.count, self.name, value.data)
        else:
            instance.add_train_value(self.name, value)


class FileWriterMeta(type):
    """Constructs writer class"""
    def __new__(cls, name, bases, attrs):
        attr_meta = attrs.pop('Meta', None)

        new_attrs = {}
        datasets = {}
        for base in reversed(bases):
            if issubclass(base, FileWriterBase):
                datasets.update(base.datasets)

        for key, val in attrs.items():
            if isinstance(val, Dataset):
                datasets[key] = val
            else:
                new_attrs[key] = val

        new_attrs['datasets'] = datasets
        new_class = super().__new__(cls, name, bases, new_attrs)

        meta = attr_meta or getattr(new_class, 'Meta', None)
        base_meta = getattr(new_class, '_meta', None)
        new_class._meta = Options(meta, base_meta)

        for ds_name, ds in datasets.items():
            ds.resolve_name(new_class._meta.aliases)
            if new_class._meta.class_attrs_interface:
                setattr(new_class, ds_name, ds.get_attribute_setter(ds_name))
            else:
                setattr(new_class, ds_name, BlockedSetter())

        return new_class


class FileWriterBase(object):
    """Writes data in EuXFEL format"""
    list_of_sources = []
    datasets = {}

    def __new__(cls, *args, **kwargs):
        if not cls.datasets:
            raise TypeError(f"Can't instantiate class {cls.__name__}, "
                            "because it has no datasets")
        return super().__new__(cls)

    def __init__(self, filename, **kwargs):
        self._ds_attrs = {}
        self._train_data = {}
        self.trains = []
        self.timestamp = []
        self.flags = []
        self.seq = 0
        self.filename = filename
        self.param = kwargs

        sources = {}
        for ds_name, ds in self.datasets.items():
            ds.init_dataset_attr(self)
            ds_attr = ds(self)
            sources.setdefault(ds_attr.source_name, ds_attr.stype)

        self.list_of_sources = list(
            (Source.SECTION[src_type], src_name)
            for src_name, src_type in sources.items()
        )

        self.nsource = len(self.list_of_sources)
        self.sources = {}
        self.source_ntrain = {}
        for sect, src_name in self.list_of_sources:
            stype = Source.SECTION.index(sect)
            self.sources[src_name] = Source(self, src_name, stype)
            self.source_ntrain[src_name] = 0

        for dsname, ds in self.datasets.items():
            src_name = ds(self).source_name
            self.sources[src_name].add(dsname, ds)

        file = h5py.File(filename.format(seq=self.seq), 'w')
        try:
            self.init_file(file)
        except Exception as e:
            file.close()
            raise e

    def init_file(self, file):
        """Initialises a new file"""
        self._file = file
        self.write_metadata()
        self.create_indices()
        self.create_datasets()

    def close(self):
        """Finalises writing and close a file"""
        self.rotate_sequence_file(True)
        self.close_datasets()
        self.write_indices()
        self._file.close()

    def write_metadata(self):
        """Write the METADATA section, including lists of sources"""
        from . import __version__
        vlen_bytes = h5py.special_dtype(vlen=bytes)  # HDF5 vlen string, ASCII

        meta_grp = self._file.create_group('METADATA')
        meta_grp.create_dataset('dataFormatVersion', dtype=vlen_bytes,
                                data=['1.0'])
        meta_grp.create_dataset('daqLibrary', dtype=vlen_bytes,
                                data=[f'EXtra-data {__version__}'])
        # TODO?: creationDate, karaboFramework, proposalNumber, runNumber,
        #  sequenceNumber, updateDate

        sources_grp = meta_grp.create_group('dataSources')
        sources_grp.create_dataset('dataSourceId', dtype=vlen_bytes, data=[
            sect + '/' + src for sect, src in self.list_of_sources
        ])

        sections, sources = (zip(*self.list_of_sources)
                             if self.nsource else (None, None))
        sources_grp.create_dataset('root', dtype=vlen_bytes, data=sections)
        sources_grp.create_dataset('deviceId', dtype=vlen_bytes, data=sources)

    def create_indices(self):
        """Creates and allocate the datasets for indices in the file
        but doesn't write real data"""
        max_trains = self._meta.max_train_per_file
        index_datasets = [
            ('trainId', np.uint64),
            ('timestamp', np.uint64),
            ('flag', np.uint32),
        ]
        self.index_grp = self._file.create_group('INDEX')
        for key, dtype in index_datasets:
            ds = self.index_grp.create_dataset(
                key, (max_trains,), dtype=dtype, chunks=(max_trains,),
                maxshape=(None,)
            )
            ds[:] = 0

        for sname, src in self.sources.items():
            src.create_index(self.index_grp, max_trains)

    def write_indices(self):
        """Write real indices to the file"""
        ntrains = len(self.trains)
        if self._meta.break_into_sequence:
            ntrains = min(self._meta.max_train_per_file, ntrains)
        index_datasets = [
            ('trainId', self.trains),
            ('timestamp', self.timestamp),
            ('flag', self.flags),
        ]
        for key, data in index_datasets:
            ds = self.index_grp[key]
            ds.resize(ntrains, 0)
            ds[:] = data[:ntrains]

        for src_name, src in self.sources.items():
            src.write_index(self.index_grp, ntrains)
            self.source_ntrain[src_name] = src.get_min_trains()

        del self.trains[:ntrains]
        del self.timestamp[:ntrains]
        del self.flags[:ntrains]

    def create_datasets(self):
        """Creates datasets in the file"""
        for sname, src in self.sources.items():
            src.create()

    def close_datasets(self):
        """Writes rest of buffered data in datasets and set final size"""
        for src in self.sources.values():
            src.close_datasets()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

    def add_value(self, count, name, value):
        """Fills a single dataset across multiple trains"""
        ds = self.datasets[name]

        # check shape value
        nrec = ds.check_value(self, value)
        ntrain = np.size(count)
        if nrec != np.sum(count):
            raise ValueError("total counts is not equal to "
                             "the number of records")

        # check count
        src_name = ds(self).source_name
        src = self.sources[src_name]
        if src.get_ntrain(name) + ntrain > len(self.trains):
            raise ValueError("the number of trains in this data exeeds "
                             "the number of trains in file")

        src.add_data(np.array(count), name, value)

        self.source_ntrain[src_name] = src.get_min_trains()
        self.rotate_sequence_file()

    def add_train_value(self, name, value):
        """Fills a single dataset in the current train"""
        ds = self.datasets[name]
        nrec = ds.check_value(self, value)

        # check count
        src_name = ds(self).source_name
        src = self.sources[src_name]
        if src.get_ntrain(name) + 1 > len(self.trains):
            raise ValueError("the number of trains in this data exeeds "
                             "the number of trains in file")

        src.add_train_data(nrec, name, value)

        self.source_ntrain[src_name] = src.get_min_trains()
        self.rotate_sequence_file()

    def add_data(self, count, **kwargs):
        """Adds data"""
        for name, value in kwargs.items():
            self.add_value(count, name, value)

    def add_train_data(self, **kwargs):
        """Adds data to the current train"""
        for name, value in kwargs.items():
            self.add_train_value(name, value)

    def rotate_sequence_file(self, finalize=False):
        """opens a new sequence file if necessary"""
        if self._meta.break_into_sequence and self.nsource:
            op = (lambda a: max(a)) if finalize else (lambda a: min(a))
            ntrain = op(self.source_ntrain.values())
            while ntrain > self._meta.max_train_per_file:

                self.close_datasets()
                self.write_indices()
                self._file.close()

                self.seq += 1

                file = h5py.File(self.filename.format(seq=self.seq), 'w')
                self.init_file(file)

                ntrain = op(self.source_ntrain.values())

    def add_trains(self, tid, ts):
        """Adds trains to the file"""
        ntrain = len(tid)
        if ntrain != len(ts):
            raise ValueError("arguments must have the same size")

        self.trains += tid
        self.timestamp += ts
        self.flags += [1] * ntrain

    def add_train(self, tid, ts):
        self.trains.append(tid)
        self.timestamp.append(ts)
        self.flags.append(1)


DS = Dataset
trs_ = MultiTrainData


class FileWriter(FileWriterBase, metaclass=FileWriterMeta):
    """Writes data into European XFEL file format

    Create a new class inherited from :class:`FileWriter`
    and use :class:`DS` to declare datasets:

    .. code-block:: python

        ctrl_grp = 'MID_DET_AGIPD1M-1/x/y'
        inst_grp = 'MID_DET_AGIPD1M-1/x/y:output'
        nbin = 1000

        class MyFileWriter(FileWriter):
            gv = DS(ctrl_grp, 'geom.fragmentVectors', (10,100), float)
            nb = DS(ctrl_grp, 'param.numberOfBins', (), np.uint64)
            rlim = DS(ctrl_grp, 'param.radiusRange', (2,), float)

            tid = DS(inst_grp, 'azimuthal.trainId', (), np.uint64)
            pid = DS(inst_grp, 'azimuthal.pulseId', (), np.uint64)
            v = DS(inst_grp, 'azimuthal.profile', (nbin,), float)

            class Meta:
                max_train_per_file = 10
                break_into_sequence = True

    Subclass :class:`Meta` is a special class for options.

    Use new class to write data in files by trains:

    .. code-block:: python

        filename = 'mydata-{seq:03d}.h5'
        with MyFileWriter(filename) as wr:
            # add data (funcion kwargs interface)
            wr.add(gv=gv, nb=nbin, rlim=(0.003, 0.016))

            for tid in trains:
                # create/compute data
                v = np.random.randn(npulse, nbin)
                vref.append(v)
                # add data (class attribute interface)
                wr.tid = [tid] * npulse
                wr.pid = pulses
                wr.v = v
                # write train
                wr.write_train(tid, 0)

    For the sources in 'CONTROL' section, the last added data repeats in
    the following trains. Only one entry is allowed per train in this section.

    For the sources in 'INSTRUMENT' section, data is dropped after flushing.
    One train may contain multiple entries. The number of entries may vary
    from train to train. All datasets in one source must have the same number
    of entries in the same train.
    """
    @classmethod
    def open(cls, fn, datasets, **kwargs):
        class_name = cls.__name__ + '_' + str(id(datasets))

        aliases = kwargs.get('aliases', {})
        attrs = {}
        for name, val in datasets.items():
            if isinstance(val, dict):
                for ds_name, ds in val.items():
                    if isinstance(ds, Dataset):
                        isalias, src_id, key = ds.orig_name
                        src_suffix = aliases[src_id] if isalias else src_id
                        new_name = (name + '/' + src_suffix
                                    if src_suffix else name)
                        ds.set_name(new_name, key)
                    attrs[ds_name] = ds
            else:
                attrs[name] = val

        if kwargs:
            attrs['Meta'] = type(class_name + '.Meta', (object,), kwargs)

        newcls = type(class_name, (cls,), attrs)
        return newcls(fn)
