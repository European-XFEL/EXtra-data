"""Writing CXI files from AGIPD/LPD data"""
import h5py
import logging
import numpy as np

log = logging.getLogger(__name__)


class VirtualCXIWriter:
    """Machinery to write a CXI file containing virtual datasets.

    You don't normally need to use this class directly. Instead,
    use the write_virtual_cxi() method on an AGIPD/LPD data interface object.

    CXI specifies a particular layout of data in the HDF5 file format.
    It is documented here:
    http://www.cxidb.org/cxi.html

    This code writes version 1.5 CXI files.

    Parameters
    ----------

    detdata: extra_data.components.MPxDetectorBase
      The detector data interface for the data to gather in this file.
    """
    def __init__(self, detdata):
        self.detdata = detdata

        self.modulenos = sorted(detdata.modno_to_source)

        frame_counts = detdata.frame_counts
        self.nframes = frame_counts.sum()
        log.info("Up to %d frames per train, %d frames in total",
                 frame_counts.max(), self.nframes)

        self.train_ids_perframe = np.repeat(
            frame_counts.index.values, frame_counts.values.astype(np.intp)
        )

        # cumulative sum gives the end of each train, subtract to get start
        self.train_id_to_ix = frame_counts.cumsum() - frame_counts

    @property
    def nmodules(self):
        return self.detdata.n_modules

    @property
    def data(self):
        return self.detdata.data

    def collect_pulse_ids(self):
        # Gather pulse IDs
        NO_PULSE_ID = 9999
        pulse_ids = np.full((self.nframes, self.nmodules), NO_PULSE_ID,
                            dtype=np.uint64)

        for source, modno in self.detdata.source_to_modno.items():
            for chunk in self.data._find_data_chunks(source, 'image.pulseId'):
                # In some cases, there's an extra dimension of length 1
                if chunk.dataset.ndim > 1:
                    chunk_data = chunk.dataset[:, 0]
                else:
                    chunk_data = chunk.dataset
                self._map_chunk(chunk, chunk_data, pulse_ids, modno)

        # Sanity checks on pulse IDs
        pulse_ids_min = pulse_ids.min(axis=1)
        if (pulse_ids_min == NO_PULSE_ID).any():
            raise Exception("Failed to find pulse IDs for some data")
        pulse_ids[pulse_ids == NO_PULSE_ID] = 0
        if (pulse_ids_min != pulse_ids.max(axis=1)).any():
            raise Exception("Inconsistent pulse IDs for different modules")

        # Pulse IDs make sense. Drop the modules dimension, giving one pulse ID
        # for each frame.
        return pulse_ids_min

    def _map_chunk(self, chunk, chunk_data, target, tgt_ax1, have_data=None):
        """Map data from chunk into target

        chunk points to contiguous source data, but if this misses a train,
        it might not correspond to a contiguous region in the output. So this
        may perform multiple mappings.
        """
        # Expand the list of train IDs to one per frame
        chunk_tids = np.repeat(chunk.train_ids, chunk.counts.astype(np.intp))

        chunk_match_start = int(chunk.first)

        while chunk_tids.size > 0:
            # Look up where the start of this chunk fits in the target
            tgt_start = int(self.train_id_to_ix[chunk_tids[0]])

            target_tids = self.train_ids_perframe[tgt_start : tgt_start+len(chunk_tids)]
            assert target_tids.shape == chunk_tids.shape, \
                f"{target_tids.shape} != {chunk_tids.shape}"
            assert target_tids[0] == chunk_tids[0], \
                f"{target_tids[0]} != {chunk_tids[0]}"

            # How much of this chunk can be mapped in one go?
            mismatches = (chunk_tids != target_tids).nonzero()[0]
            if mismatches.size > 0:
                n_match = mismatches[0]
            else:
                n_match = len(chunk_tids)

            # Select the matching data and add it to the target
            chunk_match_end = chunk_match_start + n_match
            matched = chunk_data[chunk_match_start:chunk_match_end]
            target[tgt_start : tgt_start+n_match, tgt_ax1] = matched

            # Fill in the map of what data we have
            if have_data is not None:
                have_data[tgt_start : tgt_start+n_match, tgt_ax1] = True

            # Prepare remaining data in the chunk for the next match
            chunk_match_start = chunk_match_end
            chunk_tids = chunk_tids[n_match:]

    def collect_data(self):
        src = next(iter(self.detdata.source_to_modno))
        h5file = self.data._source_index[src][0].file
        image_grp = h5file['INSTRUMENT'][src]['image']

        VLayout = h5py.VirtualLayout

        if 'gain' in image_grp:
            log.info("Identified calibrated data")
            shape = (self.nframes, self.nmodules) + self.detdata.module_shape
            log.info("Virtual data shape: %r", shape)

            layouts = {
                'data': VLayout(shape, dtype=image_grp['data'].dtype),
                'gain': VLayout(shape, dtype=image_grp['gain'].dtype),
            }

            if 'mask' in image_grp:
                layouts['mask'] = VLayout(shape, dtype=image_grp['mask'].dtype)
        else:
            log.info("Identified raw data")

            shape = (self.nframes, self.nmodules) + image_grp['data'].shape[1:]
            log.info("Virtual data shape: %r", shape)

            layouts = {
                'data': VLayout(shape, dtype=image_grp['data'].dtype),
            }

        layouts['cellId'] = VLayout((self.nframes, self.nmodules),
                                    dtype=image_grp['cellId'].dtype)

        for name, layout in layouts.items():
            key = 'image.{}'.format(name)
            have_data = np.zeros((self.nframes, self.nmodules), dtype=bool)

            for source, modno in self.detdata.source_to_modno.items():
                for chunk in self.data._find_data_chunks(source, key):
                    vsrc = h5py.VirtualSource(chunk.dataset)
                    self._map_chunk(chunk, vsrc, layout, modno, have_data)

            filled_pct = 100 * have_data.sum() / have_data.size
            log.info("Assembled %d chunks for %s, filling %.2f%% of the hyperslab",
                     len(layout.sources), key, filled_pct)

        return layouts

    def write(self, filename, fillvalues=None):
        """Write the file on disc to filename

        Parameters
        ----------
        filename: str
            Path of the file to be written.
        fillvalues: dict, optional
            keys are datasets names (one of: data, gain, mask) and associated
            fill value for missing data. defaults are:
            - data: nan (proc, float32) or 0 (raw, uint16)
            - gain: 0 (uint8)
            - mask: 0xffffffff (uint32)
        """
        pulse_ids = self.collect_pulse_ids()
        experiment_ids = np.core.defchararray.add(np.core.defchararray.add(
            self.train_ids_perframe.astype(str), ':'), pulse_ids.astype(str))

        layouts = self.collect_data()

        _fillvalues = {
            # data can be uint16 (raw) or float32 (proc)
            'data': np.nan if layouts['data'].dtype.kind == 'f' else 0,
            'gain': 0,
            'mask': 0xffffffff
        }
        if fillvalues:
            _fillvalues.update(fillvalues)
        # enforce that fill values are compatible with array dtype
        _fillvalues['data'] = layouts['data'].dtype.type(_fillvalues['data'])
        if 'gain' in layouts:
            _fillvalues['gain'] = layouts['gain'].dtype.type(_fillvalues['gain'])
        if 'mask' in layouts:
            _fillvalues['mask'] = layouts['mask'].dtype.type(_fillvalues['mask'])

        log.info("Writing to %s", filename)

        # Virtual datasets require HDF5 >= 1.10. Specifying this up front should
        # mean it fails before touching the file if run on an older version.
        # We also specify this as the maximum version, to ensure we're creating
        # files that can be read by HDF5 1.10.
        with h5py.File(filename, 'w', libver=('v110', 'v110')) as f:
            f.create_dataset('cxi_version', data=[150])
            d = f.create_dataset('entry_1/experiment_identifier',
                                 shape=experiment_ids.shape,
                                 dtype=h5py.special_dtype(vlen=str))
            d[:] = experiment_ids

            # pulseId, trainId, cellId are not part of the CXI standard, but
            # it allows extra data.
            f.create_dataset('entry_1/pulseId', data=pulse_ids)
            f.create_dataset('entry_1/trainId', data=self.train_ids_perframe)
            cellids = f.create_virtual_dataset('entry_1/cellId',
                                               layouts['cellId'])
            cellids.attrs['axes'] = 'experiment_identifier:module_identifier'

            dgrp = f.create_group('entry_1/instrument_1/detector_1')
            if len(layouts['data'].shape) == 4:
                axes_s = 'experiment_identifier:module_identifier:y:x'
            else:
                # 5D dataset, with extra axis for
                axes_s = 'experiment_identifier:module_identifier:data_gain:y:x'
                ndg = layouts['data'].shape[2]
                d = f.create_dataset('entry_1/data_gain', shape=(ndg,),
                                     dtype=h5py.special_dtype(vlen=str))
                d[:] = (['data', 'gain'] if ndg == 2 else ['data'])
                dgrp['data_gain'] = h5py.SoftLink('/entry_1/data_gain')

            data = dgrp.create_virtual_dataset(
                'data', layouts['data'], fillvalue=_fillvalues['data']
            )
            data.attrs['axes'] = axes_s

            if 'gain' in layouts:
                gain = dgrp.create_virtual_dataset(
                    'gain', layouts['gain'], fillvalue=_fillvalues['gain']
                )
                gain.attrs['axes'] = axes_s

            if 'mask' in layouts:
                mask = dgrp.create_virtual_dataset(
                    'mask', layouts['mask'], fillvalue=_fillvalues['mask']
                )
                mask.attrs['axes'] = axes_s

            dgrp['experiment_identifier'] = h5py.SoftLink('/entry_1/experiment_identifier')

            f['entry_1/data_1'] = h5py.SoftLink('/entry_1/instrument_1/detector_1')

            dgrp.create_dataset('module_identifier', data=self.modulenos)

        log.info("Finished writing virtual CXI file")
