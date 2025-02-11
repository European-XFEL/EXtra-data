import numpy as np
import h5py

class DetectorModule:
    # Overridden in subclasses:
    image_dims = ()
    detector_data_size = 0

    # Set by write_file:
    ntrains = 100
    firsttrain = 10000
    chunksize = 32

    output_parts = [
        'detector',
        'header',
        'image',
        'trailer',
    ]

    def __init__(self, device_id, frames_per_train=64, raw=True,
                 channel_name='xtdf', legacy_name=None):
        self.device_id = device_id
        self._frames_per_train = frames_per_train
        if not raw:
            # Raw data has an extra dimension, used in AGIPD to separate data
            # and gain. This dimension is removed by the calibration process.
            self.image_dims = self.image_dims[1:]
        self.raw = raw
        self.channel_name = channel_name
        self.legacy_name = legacy_name

    def write_control(self, f):
        """Write the CONTROL and RUN data, and the relevant parts of INDEX"""
        pass

    @property
    def image_keys(self):
        if self.raw:
            return [
                ('data', 'u2', self.image_dims, {}),
                ('length', 'u4', (1,), {}),
                ('status', 'u2', (1,), {}),
            ]

        else:
            return [
                ('data', 'f4', self.image_dims, {}),
                ('mask', 'u4', self.image_dims, {
                    'compression': 'gzip', 'compression_opts': 1
                }),
                ('gain', 'u1', self.image_dims, {
                    'compression': 'gzip', 'compression_opts': 1
                }),
                ('length', 'u4', (1,), {}),
                ('status', 'u2', (1,), {}),
            ]

    @property
    def other_keys(self):
        return [
            ('detector/data', 'u1', (self.detector_data_size,)),
            ('header/dataId', 'u8', ()),
            ('header/linkId', 'u8', ()),
            ('header/magicNumberBegin', 'i1', (8,)),
            ('header/majorTrainFormatVersion', 'u4', ()),
            ('header/minorTrainFormatVersion', 'u4', ()),
            ('header/pulseCount', 'u8', ()),
            ('header/reserved', 'u1', (16,)),
            ('trailer/checksum', 'i1', (16,)),
            ('trailer/magicNumberEnd', 'i1', (8,)),
            ('trailer/status', 'u8', ()),
        ]

    @property
    def frames_per_train(self):
        if np.ndim(self._frames_per_train) == 0:
            return np.full(self.ntrains, self._frames_per_train, np.uint64)
        return self._frames_per_train

    def write_instrument(self, f):
        """Write the INSTRUMENT data, and the relevant parts of INDEX"""
        trainids = np.arange(self.firsttrain, self.firsttrain + self.ntrains)

        ntrains_pad = self.ntrains
        if ntrains_pad % self.chunksize:
            ntrains_pad += + self.chunksize - (ntrains_pad % self.chunksize)

        inst_source = f'{self.device_id}:{self.channel_name}'

        # INDEX
        for part in self.output_parts:
            i_first = f.create_dataset(f'INDEX/{inst_source}/{part}/first',
                                       (self.ntrains,), 'u8', maxshape=(None,))
            i_count = f.create_dataset(f'INDEX/{inst_source}/{part}/count',
                                       (self.ntrains,), 'u8', maxshape=(None,))
            if part == 'image':
                # First first is always 0
                i_first[1:] = np.cumsum(self.frames_per_train)[:-1]
                i_count[:] = self.frames_per_train
            else:
                i_first[:] = np.arange(self.ntrains)
                i_count[:] = 1


        # INSTRUMENT (image)
        nframes = self.frames_per_train.sum()

        tid_index = np.repeat(trainids, self.frames_per_train.astype(np.intp))
        pid_index = np.concatenate([
            np.arange(0, n, dtype='u8') for n in self.frames_per_train
        ])
        if self.raw:
            # Raw data have an extra dimension (length 1) and an unlimited max
            # for the first dimension.
            ds = f.create_dataset(f'INSTRUMENT/{inst_source}/image/trainId',
                                  (nframes, 1), 'u8', maxshape=(None, 1))
            ds[:, 0] = tid_index

            pid = f.create_dataset(f'INSTRUMENT/{inst_source}/image/pulseId',
                                   (nframes, 1), 'u8', maxshape=(None, 1))
            pid[:, 0] = pid_index

            cid = f.create_dataset(f'INSTRUMENT/{inst_source}/image/cellId',
                                   (nframes, 1), 'u2', maxshape=(None, 1))
            cid[:, 0] = pid_index  # Cell IDs mirror pulse IDs for now
        else:
            # Corrected data drops the extra dimension, and maxshape==shape.
            f.create_dataset(
                f'INSTRUMENT/{inst_source}/image/trainId',
                (nframes,), 'u8', chunks=True, data=tid_index
            )

            f.create_dataset(
                f'INSTRUMENT/{inst_source}/image/pulseId',
                (nframes,), 'u8', chunks=True, data=pid_index
            )

            f.create_dataset(  # Cell IDs mirror pulse IDs for now
                f'INSTRUMENT/{inst_source}/image/cellId',
                (nframes,), 'u2', chunks=True, data=pid_index
            )

        max_len = None if self.raw else nframes
        for (key, datatype, dims, kw) in self.image_keys:
            if dims == self.image_dims and 'chunks' not in kw:
                kw['chunks'] = (1,) + dims
            f.create_dataset(
                f'INSTRUMENT/{inst_source}/image/{key}',
                shape=(nframes,) + dims,
                dtype=datatype,
                maxshape=((max_len,) + dims),
                **kw
            )


        # INSTRUMENT (other parts)
        for part in ['detector', 'header', 'trailer']:
            ds = f.create_dataset(f'INSTRUMENT/{inst_source}/{part}/trainId',
                                  (ntrains_pad,), 'u8', maxshape=(None,))
            ds[:self.ntrains] = trainids

        for (key, datatype, dims) in self.other_keys:
            f.create_dataset(f'INSTRUMENT/{inst_source}/{key}',
                     (ntrains_pad,) + dims, datatype, maxshape=((None,) + dims))

        if self.legacy_name is not None:
            # The legacy source name for corrected data is the same as for
            # raw data, which for these detectors always has the xtdf channel.
            f[f'INDEX/{self.legacy_name}:xtdf'] = h5py.SoftLink(
                f'/INDEX/{inst_source}')
            f[f'INSTRUMENT/{self.legacy_name}:xtdf'] = h5py.SoftLink(
                f'/INSTRUMENT/{inst_source}')

    def datasource_ids(self):
        for part in self.output_parts:
            yield f'INSTRUMENT/{self.device_id}:{self.channel_name}/{part}'

        if self.legacy_name is not None:
            for part in self.output_parts:
                yield f'INSTRUMENT/{self.legacy_name}:xtdf/{part}'


class AGIPDModule(DetectorModule):
    image_dims = (2, 512, 128)
    detector_data_size = 5408

class LPDModule(DetectorModule):
    image_dims = (1, 256, 256)
    detector_data_size = 416

class DSSCModule(DetectorModule):
    image_dims = (1, 128, 512)
    detector_data_size = 416
