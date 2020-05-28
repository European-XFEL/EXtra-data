import h5py
import numpy as np


class FileWriter:
    """Write data in European XFEL HDF5 format

    This is intended to allow copying a subset of data into a smaller,
    more portable file.
    """

    def __init__(self, path, data):
        self.file = h5py.File(path, 'w')
        self.data = data
        self.indexes = {}  # {path: (first, count)}
        self.data_sources = set()

    def prepare_source(self, source):
        """Prepare all the datasets for one source.

        We do this as a separate step so the contents of the file are defined
        together before the main data.
        """
        for key in sorted(self.data.keys_for_source(source)):
            path = f"{self._section(source)}/{source}/{key.replace('.', '/')}"
            nentries = self._guess_number_of_storing_entries(source, key)
            src_ds1 = self.data._source_index[source][0].file[path]
            self.file.create_dataset_like(
                path, src_ds1, shape=(nentries,) + src_ds1.shape[1:],
            )
            if source in self.data.instrument_sources:
                self.data_sources.add(f"INSTRUMENT/{source}/{key.partition('.')[0]}")

        if source not in self.data.instrument_sources:
            self.data_sources.add(f"CONTROL/{source}")

    def _guess_number_of_storing_entries(self, source, key):
        """Provide the length for the initial dataset to create.

        May be overridden in subclasses.
        """
        return self.data.get_data_counts(source, key).sum()

    def _section(self, source):
        if source in self.data.instrument_sources:
            return 'INSTRUMENT'
        else:
            return 'CONTROL'

    def copy_dataset(self, source, key):
        """Copy data into a dataset"""
        a = self.data.get_array(source, key)
        path = f"{self._section(source)}/{source}/{key.replace('.', '/')}"
        self.file[path][:] = a.values
        self._make_index(source, key, a.coords['trainId'].values)

    def _make_index(self, source, key, data_tids):
        # Original files contain exactly 1 entry per train for control data,
        # but if one file starts before another, there can be some values
        # missing when we collect several files together. We don't try to
        # extrapolate to fill missing data, so some counts may be 0.

        if source in self.data.instrument_sources:
            index_path = source + '/' + key.partition('.')[0]
        else:
            index_path = source

        if index_path not in self.indexes:
            if source not in self.data.instrument_sources:
                assert len(np.unique(data_tids)) == len(data_tids),\
                    "Duplicate train IDs in control data!"

            self.indexes[index_path] = self._generate_index(data_tids)

    def _generate_index(self, data_tids):
        """Convert an array of train IDs to first/count for each train"""
        assert (np.diff(data_tids) >= 0).all(), "Out-of-order train IDs"
        counts = np.array([np.count_nonzero(t == data_tids)
                          for t in self.data.train_ids], dtype=np.uint64)
        firsts = np.zeros_like(counts)
        firsts[1:] = np.cumsum(counts)[:-1]  # firsts[0] is always 0

        return firsts, counts

    def copy_source(self, source):
        """Copy data for all keys of one source"""
        for key in self.data.keys_for_source(source):
            self.copy_dataset(source, key)

    def write_train_ids(self):
        self.file.create_dataset(
            'INDEX/trainId', data=self.data.train_ids, dtype='u8'
        )

    def write_indexes(self):
        """Write the INDEX information for all data we've copied"""
        for groupname, (first, count) in self.indexes.items():
            group = self.file.create_group(f'INDEX/{groupname}')
            group.create_dataset('first', data=first, dtype=np.uint64)
            group.create_dataset('count', data=count, dtype=np.uint64)

    def write_metadata(self):
        """Write the METADATA section, including lists of sources"""
        vlen_bytes = h5py.special_dtype(vlen=bytes)
        data_sources = sorted(self.data_sources)
        N = len(data_sources)

        sources_ds = self.file.create_dataset(
            'METADATA/dataSourceId', (N,), dtype=vlen_bytes, maxshape=(None,)
        )
        sources_ds[:] = data_sources

        root_ds = self.file.create_dataset(
            'METADATA/root', (N,), dtype=vlen_bytes, maxshape=(None,)
        )
        root_ds[:] = [ds.split('/', 1)[0] for ds in data_sources]

        devices_ds = self.file.create_dataset(
            'METADATA/deviceId', (N,), dtype=vlen_bytes, maxshape=(None,)
        )
        devices_ds[:] = [ds.split('/', 1)[1] for ds in data_sources]

    def set_writer(self):
        """Record the package & version writing the file in an attribute"""
        from . import __version__

        self.file.attrs['writer'] = 'extra_data {}'.format(__version__)

    def write(self):
        d = self.data
        self.set_writer()
        self.write_train_ids()

        for source in d.all_sources:
            self.prepare_source(source)

        self.write_metadata()

        for source in d.all_sources:
            self.copy_source(source)

        self.write_indexes()


class VirtualFileWriter(FileWriter):
    """Write virtual datasets in European XFEL format

    The new files refer to the original data files, so they aren't portable,
    but they provide more convenient access by reassembling data spread over
    several sequence files.
    """
    def __init__(self, path, data):
        if not hasattr(h5py, 'VirtualLayout'):
            raise Exception("Creating virtual datasets requires HDF5 1.10 "
                            "and h5py 2.9")

        super().__init__(path,  data)

    def _assemble_data(self, source, key):
        """Assemble chunks of data into a virtual layout"""
        # First, get a list of all non-empty data chunks
        chunks = [c for c in self.data._find_data_chunks(source, key)
                  if (c.counts > 0).any()]
        chunks.sort(key = lambda c: c.train_ids[0])
        if not chunks:
            return None, None

        # Create the layout, which will describe what data is where
        n_total = np.sum([c.counts.sum() for c in chunks])
        ds0 = chunks[0].dataset
        layout = h5py.VirtualLayout(shape=(n_total,) + ds0.shape[1:],
                                    dtype=ds0.dtype)

        # Map each chunk into the relevant part of the layout
        output_cursor = np.uint64(0)
        for chunk in chunks:
            n = chunk.counts.sum()
            src = h5py.VirtualSource(chunk.dataset)
            src = src[chunk.slice]
            layout[output_cursor : output_cursor + n] = src
            output_cursor += n

        assert output_cursor == n_total

        # Make an array of which train ID each data entry is for:
        train_ids = np.concatenate([
            np.repeat(c.train_ids, c.counts.astype(np.intp)) for c in chunks
        ])
        return layout, train_ids

    def prepare_source(self, source):
        for key in self.data.keys_for_source(source):
            self.add_dataset(source, key)

    def add_dataset(self, source, key):
        layout, train_ids = self._assemble_data(source, key)
        if not layout:
            return  # No data

        path = f"{self._section(source)}/{source}/{key.replace('.', '/')}"
        self.file.create_virtual_dataset(path, layout)

        self._make_index(source, key, train_ids)
        if source in self.data.instrument_sources:
            self.data_sources.add(f"INSTRUMENT/{source}/{key.partition('.')[0]}")
        else:
            self.data_sources.add(f"CONTROL/{source}")

        return path

    def copy_source(self, source):
        pass  # Override base class copying data
