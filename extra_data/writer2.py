import h5py
import numpy as np

class DatasetDescr:
    def __init__(self, dtype, entry_shape=(), chunks=None, compression=None):
        self.dtype = dtype
        self.entry_shape = entry_shape
        self.chunks = chunks  # Guess chunk shape? Or require it specified?
        self.compression = compression


sources = {
    'SCS_DET_DSSC1M-1/DET/0CH0:xtdf': {
        'image.data': DatasetDescr(np.uint16, (128, 512)),
        'image.pulseId': DatasetDescr(np.uint16, (1,)),
        # ...
    }
}
# TODO: distinguish INSTRUMENT vs CONTROL sources?

# Reuse the same keys for several sources (this is similar to mockdata machinery for testing)
def xtdf_detector_module(shape):
    ...

sources = {
    'SCS_DET_DSSC1M-1/DET/0CH0:xtdf': xtdf_detector_module((128, 512))
}

def make_file_sources(inst_sources: dict, ctrl_sources: dict):
    res = set()
    for src in ctrl_sources:
        res.add(('CONTROL', src))
    for src, keys in inst_sources.items():
        for key in keys:
            res.add(('INSTRUMENT', f"{src}/{key.partition('.')[0]}"))
    return sorted(res)

class FileWriter:
    def __init__(self, path, inst_sources: dict, ctrl_sources: dict, max_trains: int =None):
        # Create METADATA up front, reserve space for all INDEX datasets,
        # create but don't allocate CONTROL & INSTRUMENT datasets
        self.path = path
        self.inst_sources = inst_sources
        self.ctrl_sources = ctrl_sources
        self._file_sources = make_file_sources(inst_sources, ctrl_sources)
        self.max_trains = max_trains
        self.n_trains = 0
        self.f = h5py.File(path, 'w')

        self.write_metadata()
        self.create_indices()
        self.create_datasets()

    def write_metadata(self):
        """Write the METADATA section, including lists of sources"""
        from . import __version__
        vlen_bytes = h5py.special_dtype(vlen=bytes)  # HDF5 vlen string, ASCII

        md_grp = self.f.create_group('METADATA')
        md_grp.create_dataset('dataFormatVersion', dtype=vlen_bytes, data=['1.0'])
        md_grp.create_dataset('daqLibrary', dtype=vlen_bytes, data=[
            f'EXtra-data {__version__}'
        ])
        # TODO?: creationDate, karaboFramework, proposalNumber, runNumber,
        #  sequenceNumber, updateDate

        sources_grp = self.f.create_group('METADATA/dataSources')
        sources_grp.create_dataset('dataSourceId', dtype=vlen_bytes, data=[
            f'{sect}/{src}' for (sect, src) in self._file_sources
        ])
        sources_grp.create_dataset('root', dtype=vlen_bytes, data=[
            section for (section, _) in self._file_sources
        ])
        sources_grp.create_dataset('deviceId', dtype=vlen_bytes, data=[
            src for (_, src) in self._file_sources
        ])

    def create_indices(self):
        def make_ds(grp, name, dtype=np.uint64):
            # Set the chunk size so each index dataset is just 1 chunk to read
            ds = grp.create_dataset(
                name, (self.max_trains,), dtype=dtype, chunks=(self.max_trains,)
            )
            # Write to the dataset so HDF5 allocates space for it, ensuring
            # the indices are at the beginning of the file.
            ds[0] = 0

        make_ds(self.f, 'INDEX/trainId')
        make_ds(self.f, 'INDEX/timestamp')
        make_ds(self.f, 'INDEX/flag', dtype=np.uint32)

        for _, src_name in self._file_sources:
            group = self.f.create_group(f'INDEX/{src_name}')
            make_ds(group, 'first')
            make_ds(group, 'count')

    def create_datasets(self):
        for src, keys in self.ctrl_sources.items():
            grp = self.f.create_group(f'CONTROL/{src}')
            for key, info in keys.items():
                grp.create_dataset(
                    key, (0,) + info.entry_shape, dtpe=info.dtype,
                    chunks=info.chunks, compression=info.compression
                )
        for src, keys in self.inst_sources.items():
            grp = self.f.create_group(f'INSTRUMENT/{src}')
            for key, info in keys.items():
                grp.create_dataset(
                    key, (0,) + info.entry_shape, dtpe=info.dtype,
                    chunks=info.chunks, compression=info.compression
                )

    def add_trains(self, train_ids: np.ndarray, timestamps=None):
        # Do we need to add trains incrementally? Or can we always do them up front?
        end = self.n_trains + len(train_ids)
        self.f['INDEX/trainId'][self.n_trains:end] = train_ids
        if timestamps is not None:
            self.f['INDEX/timestamp'][self.n_trains:end] = timestamps
        self.n_trains = end

    # Different # of entries per train OK
    # Splitting trains across calls OK

    def add_data(self, src: str, key: str, data: np.ndarray, train_ids: np.ndarray):
        # Check we're not reusing a train ID, resize dataset, write the new data.
        ...

    def finish(self):
        # Write out indices
        ...

with FileWriter('mydata.h5', sources, np.arange(10000, 10256)) as fw:
    for tid in range(10000, 10256):
        data = np.zeros((64, 128, 512), dtype=np.uint16)
        fw.add_data('SCS_DET_DSSC1M-1/DET/0CH0:xtdf', 'image.data', data, np.full(64, tid))

for seq in range(10):
    ...

# Wrapper for writing several sequence files with the same sources

class SequenceWriter:
    ...
