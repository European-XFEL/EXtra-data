import h5py
import numpy as np
from packaging import version

from .exceptions import MultiRunError


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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def prepare_source(self, source):
        """Prepare all the datasets for one source.

        We do this as a separate step so the contents of the file are defined
        together before the main data.
        """
        for key in sorted(self.data.keys_for_source(source)):
            path = f"{self._section(source)}/{source}/{key.replace('.', '/')}"
            nentries = self._guess_number_of_storing_entries(source, key)
            src_ds1 = self.data[source].files[0].file[path]
            self.file.create_dataset_like(
                path, src_ds1, shape=(nentries,) + src_ds1.shape[1:],
                # Corrected detector data has maxshape==shape, but if any max
                # dim is smaller than the chunk size, h5py complains. Making
                # the first dimension unlimited avoids this.
                maxshape=(None,) + src_ds1.shape[1:],
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
        train_timestamps = self.data.train_timestamps()
        if not np.all(np.isnat(train_timestamps)):
            self.file.create_dataset(
                'INDEX/timestamp', data=train_timestamps.astype(np.uint64)
            )

    def write_indexes(self):
        """Write the INDEX information for all data we've copied"""
        for groupname, (first, count) in self.indexes.items():
            group = self.file.create_group(f'INDEX/{groupname}')
            group.create_dataset('first', data=first, dtype=np.uint64)
            group.create_dataset('count', data=count, dtype=np.uint64)

    def write_metadata(self):
        try:
            metadata = self.data.run_metadata()
        except MultiRunError:
            metadata = {}

        metadata_grp = self.file.create_group('METADATA')
        format_version = version.parse(metadata.get('dataFormatVersion'))
        if format_version >= version.parse("1.0"):
            # We don't care about the differences between version 1.0/1.1/1.2,
            # so for simplicity we stick to the 1.0 format.
            metadata["dataFormatVersion"] = "1.0"

            self.write_sources(metadata_grp.create_group('dataSources'))

            # File format 1.0 should also have INDEX/flag
            self.file.create_dataset('INDEX/flag', data=self.gather_flag())

            for key, val in metadata.items():
                metadata_grp[key] = [val]
        else:
            # File format '0.5': source lists directly in METADATA
            self.write_sources(metadata_grp)

    def write_sources(self, data_sources_grp: h5py.Group):
        """Write the METADATA section, including lists of sources"""
        vlen_bytes = h5py.special_dtype(vlen=bytes)
        data_sources = sorted(self.data_sources)
        N = len(data_sources)

        sources_ds = data_sources_grp.create_dataset(
            'dataSourceId', (N,), dtype=vlen_bytes, maxshape=(None,)
        )
        sources_ds[:] = data_sources

        root_ds = data_sources_grp.create_dataset(
            'root', (N,), dtype=vlen_bytes, maxshape=(None,)
        )
        root_ds[:] = [ds.split('/', 1)[0] for ds in data_sources]

        devices_ds = data_sources_grp.create_dataset(
            'deviceId', (N,), dtype=vlen_bytes, maxshape=(None,)
        )
        devices_ds[:] = [ds.split('/', 1)[1] for ds in data_sources]

    def gather_flag(self):
        """Make the array for INDEX/flag.

        Trains are valid (1) if they are valid in *any* of the source files.
        """
        tid_arr = np.asarray(self.data.train_ids, dtype=np.uint64)
        flag = np.zeros_like(tid_arr, dtype=np.int32)
        for fa in self.data.files:
            mask_valid = np.isin(tid_arr, fa.valid_train_ids)
            flag[mask_valid] = 1

        return flag

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

    def _assemble_data(self, keydata):
        """Assemble chunks of data into a virtual layout"""
        # Create the layout, which will describe what data is where
        layout = h5py.VirtualLayout(shape=keydata.shape, dtype=keydata.dtype)

        # Map each chunk into the relevant part of the layout
        output_cursor = 0
        for chunk in keydata._data_chunks_nonempty:
            n = chunk.total_count
            src = h5py.VirtualSource(chunk.dataset)
            layout[output_cursor : output_cursor + n] = src[chunk.slice]
            output_cursor += n

        assert output_cursor == layout.shape[0]

        return layout

    # In big detector data, these fields are like extra indexes.
    # So we'll copy them to the output file for fast access, rather than
    # making virtual datasets.
    copy_keys = {'image.pulseId', 'image.cellId'}

    def prepare_source(self, source):
        srcdata = self.data[source]
        grp_out = self.file.require_group(f'{srcdata.section}/{source}')
        grp_out.attrs['source_files'] = sorted([f.filename for f in srcdata.files])

        for key in srcdata.keys():
            if key in self.copy_keys:
                self.copy_dataset(source, key)
            else:
                self.add_dataset(source, key)

        # Add a link in RUN for control sources
        if srcdata.is_control:
            src_file = srcdata.files[0]
            run_path = f'RUN/{source}'
            self.file[run_path] = h5py.ExternalLink(src_file.filename, run_path)

    def copy_dataset(self, source, key):
        """Copy data as a new dataset"""
        a = self.data.get_array(source, key)
        path = f"{self._section(source)}/{source}/{key.replace('.', '/')}"
        self.file.create_dataset(path, data=a.values, compression='gzip')
        self._make_index(source, key, a.coords['trainId'].values)

    def add_dataset(self, source, key):
        keydata = self.data[source, key]

        if keydata.shape[0] == 0:  # No data
            # Make the dataset virtual even with no source data to map.
            # This workaround will hopefully become unnecessary from h5py 3.14
            parent_path, name = keydata.hdf5_data_path.rsplit('/', 1)
            group = self.file.require_group(parent_path)

            dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
            dcpl.set_layout(h5py.h5d.VIRTUAL)

            h5py.h5d.create(
                group.id,
                name=name.encode(),
                tid=h5py.h5t.py_create(keydata.dtype, logical=1),
                space=h5py.h5s.create_simple(keydata.shape),
                dcpl=dcpl
            )
        else:
            layout = self._assemble_data(keydata)
            self.file.create_virtual_dataset(keydata.hdf5_data_path, layout)

        self._make_index(source, key, keydata.train_id_coordinates())
        if source in self.data.instrument_sources:
            self.data_sources.add(f"INSTRUMENT/{source}/{key.partition('.')[0]}")
        else:
            self.data_sources.add(f"CONTROL/{source}")

        return keydata.hdf5_data_path

    def copy_source(self, source):
        pass  # Override base class copying data
