import warnings
import h5py
import numpy as np

# compatibility to future numpy features
from .numpy_future import add_future_function_into
add_future_function_into(np)


class DatasetDescriptor:
    """Create datasets and fill the with data"""
    def __init__(self, source_name, key, entry_shape, dtype,
                 chunks=None, compression=None):
        self.entry_shape = tuple(entry_shape)
        self.dtype = np.dtype(dtype)
        self.compression = compression
        self.chunks = chunks

        # can we really distinguish sources by colons?
        self.stype = int(':' in source_name)
        if self.stype:
            tk, self.key = key.split('.', 1)
            self.source_name = source_name + '/' + tk
        else:
            self.source_name = source_name
            self.key = key

    def chunks_autosize(self, max_trains):
        if self.chunks is not None:
            return

        MN = (max_trains, 32, 32)
        SZ = (2**14, 2**19, 2**23)  # 16K, 512K, 8M

        size = np.prod(self.entry_shape)
        ndim = len(self.entry_shape)
        nbytes = size * self.dtype.itemsize

        entry_type = int(size != 1) * (1 + int(ndim > 1))
        chunk = max(SZ[entry_type] // nbytes, MN[entry_type])
        if self.stype == 0:
            chunk = min(chunk, max_trains)

        self.chunks = (chunk,) + self.entry_shape

    def create(self, grp):
        self._ds = grp.create_dataset(
            self.key.replace('.', '/'), self.chunks[:1] + self.entry_shape,
            dtype=self.dtype, chunks=self.chunks,
            maxshape=(None,) + self.entry_shape, compression=self.compression
        )
        return self._ds

    def write(self, data, pos, nrec):
        end = pos + nrec
        size = self._ds.shape[0]
        if size < end:
            self._ds.resize(size + self.chunks[0], 0)
        self._ds[pos:end] = data

    def resize(self, nrec):
        self._ds.resize(nrec, 0)


DS = DatasetDescriptor


class Source:
    """Creates data source group and its indexes"""

    SECTION = ('CONTROL', 'INSTRUMENT')

    def __init__(self, name, stype=None):
        self.name = name
        if stype is None:
            self.stype = int(':' in name)
        else:
            self.stype = stype

        self.section = self.SECTION[self.stype]
        self.datasets = []

        self.first = []
        self.count = []
        self.pos = 0
        self.nrec = 0

    def add(self, ds):
        self.datasets.append(ds)

    def create(self, file):
        grp = file.create_group(self.section + '/' + self.name)
        for ds in self.datasets:
            ds.create(grp)
        self._grp = grp
        return grp

    def create_index(self, index_grp, max_trains):
        grp = index_grp.create_group(self.name)
        for key in ('first', 'count'):
            ds = grp.create_dataset(
                key, (max_trains,), dtype=np.uint64, chunks=(max_trains,),
                maxshape=(None,)
            )
            ds[:] = 0

    def write_index(self, index_grp, ntrains):
        grp = index_grp[self.name]
        for dsname in ('first', 'count'):
            ds = grp[dsname]
            ds.resize(ntrains, axis=0)
            val = getattr(self, dsname)
            ds[:] = val[:ntrains]
            setattr(self, dsname, val[ntrains:])

        self.pos = 0
        for t in range(len(self.count)):
            self.first[t] = self.pos
            self.pos += self.count[t]

    def resize_datasets(self):
        for ds in self.datasets:
            ds.resize(self.pos)

    def write_train(self, data):
        for ds in self.datasets:
            try:
                ds.write(data[ds.key], self.pos, self.nrec)
            except KeyError:
                pass

        self.first.append(self.pos)
        self.count.append(self.nrec)

        self.pos += self.nrec
        self.nrec = int(not self.stype)

    def is_data_complete(self, data):
        if self.stype == 1 and len(data) == 0:
            return False
        missed = set(ds.key for ds in self.datasets) - set(data.keys())
        return list(missed) if missed else True


class Options:
    """Provides a set of options with overriding default values
    by ones declared in Meta subclass
    """
    NAMES = (
        'max_train_per_file', 'break_into_sequence', 'warn_on_missing_data'
    )

    def __init__(self, meta):
        self.max_train_per_file = 500
        self.break_into_sequence = False
        self.warn_on_missing_data = False

        self.override_defaults(meta)

    def override_defaults(self, meta):
        if not meta:
            return
        meta_attrs = meta.__dict__.copy()
        for attr_name in meta.__dict__:
            if attr_name.startswith('_'):
                del meta_attrs[attr_name]

        for attr_name in Options.NAMES:
            if attr_name in meta_attrs:
                setattr(self, attr_name, meta_attrs.pop(attr_name))

        if meta_attrs != {}:
            raise TypeError("'class Meta' got invalid attribute(s): " +
                            ','.join(meta_attrs))


class DataSetter:
    """Overrides the setters for attributes which declared as datasets
    in order to use the assignment operation for adding data in a train
    """
    def __init__(self, name):
        self.name = name

    def __set__(self, instance, value):
        instance.add_value(self.name, value)


class FileWriterMeta(type):
    """Constructs writer class"""
    def __new__(cls, name, bases, attrs):
        attr_meta = attrs.pop('Meta', None)

        new_attrs = {}
        datasets = {}
        sources = {}
        for key, val in attrs.items():
            if isinstance(val, DatasetDescriptor):
                datasets[key] = val
                src = sources.setdefault(
                    val.source_name,
                    Source(val.source_name, val.stype)
                )
                src.add(val)
                new_attrs[key] = DataSetter(key)
            else:
                new_attrs[key] = val

        new_attrs['list_of_sources'] = list(
            (src_inst.section, src_name)
            for src_name, src_inst in sources.items()
        )
        new_attrs['sources'] = sources
        new_attrs['datasets'] = datasets

        new_class = super().__new__(cls, name, bases, new_attrs)
        meta = attr_meta or getattr(new_class, 'Meta', None)

        new_class._meta = Options(meta)

        for ds in datasets.values():
            ds.chunks_autosize(new_class._meta.max_train_per_file)

        return new_class


class FileWriterBase:
    """Writes data in EuXFEL format"""

    def __init__(self, filename):
        self._train_data = {}
        self.trains = []
        self.timestamp = []
        self.flags = []
        self.seq = 0
        self.filename = filename
        file = h5py.File(filename.format(seq=self.seq), 'w')
        self.init_file(file)

    def init_file(self, file):
        """Initialises a new file"""
        self._file = file
        self.write_metadata()
        self.create_indices()
        self.create_datasets()

    def close(self):
        """Finalises writing and close a file"""
        self.resize_datasets()
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
        sections, sources = zip(*self.list_of_sources)
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
        index_datasets = [
            ('trainId', self.trains),
            ('timestamp', self.timestamp),
            ('flag', self.flags),
        ]
        for key, data in index_datasets:
            ds = self.index_grp[key]
            ds.resize(ntrains, 0)
            ds[:] = data

        for sname, src in self.sources.items():
            src.write_index(self.index_grp, ntrains)

        self.trains = []
        self.timestamp = []
        self.flags = []

    def create_datasets(self):
        for sname, src in self.sources.items():
            src.create(self._file)

    def resize_datasets(self):
        """Resizes datasets in the file to match the number of written
        trains/pulses"""
        for src in self.sources.values():
            src.resize_datasets()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

    @staticmethod
    def __check_value(value, entry_shape, dtype):
        """checks submitted data"""
        # if not np.can_cast(np.result_type(value), dtype, casting='unsafe'):
        #     raise(TypeError(
        #         f"invalid type: <{np.result_type(value)}> "
        #         f"cannot be cast to <{np.dtype(dtype)}>"))
        # elif not np.can_cast(value, dtype, casting='safe'):
        #     warnings.warn(
        #         f"unsafe type cast from <{np.result_type(value)}> "
        #         f"to <{np.dtype(dtype)}>", RuntimeWarning)
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

    def add_value(self, name, value):
        """Fills a single dataset in the current train"""
        ds = self.datasets[name]
        src = self.sources[ds.source_name]

        # check shape of value
        nrec = FileWriterBase.__check_value(value, ds.entry_shape, ds.dtype)
        if src.stype == 0 and nrec != 1:
            raise ValueError("shape mismatch: only one entry per train "
                             f"can be written in control source, got {nrec}")

        # all datasets in the group must have
        # the same number of records per train
        src_data = self._train_data.setdefault(src.name, {})
        nds = len(src_data)
        if src.nrec == 0:
            src.nrec = nrec
        elif (src.nrec != nrec) and (nds != 1 or ds.key not in src_data):
            raise RuntimeError(
                f"adding number of entries ({nrec}) mismatch the number "
                f"({src.nrec}) previously submitted to this source")

        # store value
        src_data[ds.key] = value

    def add(self, **kwargs):
        """Adds data to the current train"""
        for name, value in kwargs.items():
            self.add_value(name, value)

    def is_data_complete(self):
        """ Checks the completness of data"""
        missed = []
        is_complete = True
        for sname, src in self.sources.items():
            status = src.is_data_complete(self._train_data.get(sname, {}))
            if isinstance(status, list):
                missed.append((sname, status))
            else:
                is_complete = is_complete and status

        return missed if missed else is_complete

    def write_train(self, tid, ts=None):
        """Writes submitted data to the file, opens a new sequence file
        if necessary"""
        self.rotate_sequence_file()

        status = self.is_data_complete()
        if isinstance(status, list):
            raise ValueError("data was not submitted for dataset(s):\n" +
                             "\n".join(f"{src}: {key}"for src, key in status))
        elif self._meta.warn_on_missing_data and not status:
            warnings.warn("Some instruments did not submitted data "
                          f"for train {tid}", RuntimeWarning)

        for sname, src in self.sources.items():
            if sname in self._train_data:
                src.write_train(self._train_data[sname])
                if src.stype == 1:
                    del self._train_data[sname]
            else:
                src.write_train({})

        self.trains.append(tid)
        self.timestamp.append(ts)
        self.flags.append(1)

    def rotate_sequence_file(self):
        """opens a new sequence file if necessary"""
        if (self._meta.break_into_sequence and
                len(self.trains) >= self._meta.max_train_per_file):

            self.close()
            self.seq += 1

            file = h5py.File(self.filename.format(seq=self.seq), 'w')
            self.init_file(file)


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
    pass
