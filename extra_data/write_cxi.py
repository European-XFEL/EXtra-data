"""Writing CXI files from AGIPD/LPD data"""
import h5py
import logging
import numpy as np

log = logging.getLogger(__name__)


class VirtualCXIWriterBase:
    """
    Base class for machinery to write a CXI file containing virtual
    datasets.

    You don't normally need to use this class directly. Instead,
    use the write_virtual_cxi() method on a multi-module detector
    data interface object.

    CXI specifies a particular layout of data in the HDF5 file format.
    It is documented here:
    http://www.cxidb.org/cxi.html

    This code writes version 1.5 CXI files.

    Parameters
    ----------
    detdata: extra_data.components.MultimodDetectorBase
      The detector data interface for the data to gather in this file.
    """

    # 1 entry is an index along the first (time) dimension in the source files.
    # XTDF detectors (AGIPD etc.) arrange pulses along this dimension, so each
    # entry is one frame & one memory cell. JUNGFRAU in burst mode makes one
    # entry with a separate dimension for several pulses, so overrides this.
    cells_per_entry = 1

    def __init__(self, detdata):
        self.detdata = detdata
        self.group_label, self.image_label = detdata._main_data_key.split('.')

        frame_counts = detdata.frame_counts * self.cells_per_entry
        self.nframes = frame_counts.sum()
        log.info("Up to %d frames per train, %d frames in total",
                 frame_counts.max(), self.nframes)

        self.train_ids_perframe = np.repeat(
            frame_counts.index.values, frame_counts.values.astype(np.intp)
        )

        # For AGIPD, DSSC & LPD detectors modules are numbered from 0.
        # Overridden for JUNGFRAU to number from 1.
        self.modulenos = list(range(self.nmodules))

    @property
    def nmodules(self):
        """Number of detector modules."""
        return self.detdata.n_modules

    @property
    def data(self):
        """DataCollection with detector data from a run."""
        return self.detdata.data

    def _get_module_index(self, module):
        """Returns an index for the specified module."""
        return self.modulenos.index(module)

    def collect_pulse_ids(self):
        """
        Gather pulse/cell ID labels for all modules and check consistency.

        Raises
        ------
        Exception:
          Some data has no pulse ID values for any module.
        Exception:
          Inconsistent pulse IDs between detector modules.

        Returns
        -------
        pulse_ids_min: np.array
          Array of pulse IDs per frame common for all detector modules.
        """
        # Gather pulse IDs
        NO_PULSE_ID = 9999
        pulse_ids = np.full((self.nframes, self.nmodules), NO_PULSE_ID,
                            dtype=np.uint64)

        pulse_key = self.group_label + '.' + self.pulse_id_label
        for source, modno in self.detdata.source_to_modno.items():
            module_ix = self._get_module_index(modno)
            for chunk in self.data._find_data_chunks(source, pulse_key):
                chunk_data = chunk.dataset
                self._map_chunk(chunk, chunk_data, pulse_ids, module_ix)

        # Sanity checks on pulse IDs
        pulse_ids_min = pulse_ids.min(axis=1)
        if (pulse_ids_min == NO_PULSE_ID).any():
            raise Exception("Failed to find pulse IDs for some data")
        pulse_ids[pulse_ids == NO_PULSE_ID] = 0
        if (pulse_ids_min != pulse_ids.max(axis=1)).any():
            raise Exception("Inconsistent pulse IDs for different modules")

        # Pulse IDs make sense. Drop the modules dimension, giving one
        # pulse ID for each frame.
        return pulse_ids_min

    def _map_chunk(self, chunk, chunk_data, target, tgt_ax1, have_data=None):
        """
        Map data from chunk into target.

        Chunk points to contiguous source data, but if this misses a train,
        it might not correspond to a contiguous region in the output. So this
        may perform multiple mappings.

        Parameters
        ----------
        chunk: read_machinery::DataChunk
          Reference to a contiguous chunk of data to be mapped.
        chunk_data: h5py.Dataset / h5py.VirtualSource
          Dataset / VirtualSource to map data from.
        target: np.array / h5py.VirtualLayout
          Target to map data to.
        tgt_ax1: int
          Value for the target axis 1 - index corresponding to the detector
          module.
        have_data: np.array(dtype=bool), optional
          An array to monitor which part of the target have been mapped
          with data. Defaults to None.
        """
        # Expand the list of train IDs to one per frame
        for tgt_slice, chunk_slice in self.detdata._split_align_chunk(
                chunk, self.detdata.train_ids_perframe
        ):
            tgt_start = tgt_slice.start * self.cells_per_entry
            tgt_end = tgt_slice.stop * self.cells_per_entry

            if self.cells_per_entry == 1:
                # In some cases, there's an extra dimension of length 1.
                # E.g. JUNGFRAU data with 1 memory cell per train or
                # DSSC/LPD raw data.
                if (len(chunk_data.shape) > 1 and chunk_data.shape[1] == 1):
                    matched = chunk_data[chunk_slice, 0]
                else:
                    matched = chunk_data[chunk_slice]
                target[tgt_start:tgt_end, tgt_ax1] = matched
            else:
                matched = chunk_data[chunk_slice]
                if isinstance(chunk_data, h5py.VirtualSource):
                    # Use broadcasting of h5py.VirtualSource
                    target[tgt_start:tgt_end, tgt_ax1] = matched
                else:
                    target[tgt_start:tgt_end, tgt_ax1] = matched.reshape(
                        (-1,) + matched.shape[2:])

            # Fill in the map of what data we have
            if have_data is not None:
                have_data[tgt_start:tgt_end, tgt_ax1] = True

    def _map_layouts(self, layouts):
        """
        Map virtual sources into virtual layouts.

        Parameters
        ----------
        layouts: dict
          A dictionary of unmapped virtual layouts.

        Returns
        -------
        layouts: dict
          A dictionary of virtual layouts mapped to the virtual sources.
        """
        for name, layout in layouts.items():
            key = '{}.{}'.format(self.group_label, name)
            have_data = np.zeros((self.nframes, self.nmodules), dtype=bool)

            for source, modno in self.detdata.source_to_modno.items():
                print(f" ### Source: {source}, ModNo: {modno}, Key: {key}")
                module_ix = self._get_module_index(modno)
                for chunk in self.data._find_data_chunks(source, key):
                    vsrc = h5py.VirtualSource(chunk.dataset)
                    self._map_chunk(chunk, vsrc, layout, module_ix, have_data)

            filled_pct = 100 * have_data.sum() / have_data.size
            if hasattr(layout, 'sources'):
                n_mappings = len(layout.sources)  # h5py < 3.3
            else:
                n_mappings = layout.dcpl.get_virtual_count()  # h5py >= 3.3
            log.info(f"Assembled {n_mappings:d} chunks for {key:s}, "
                     f"filling {filled_pct:.2f}% of the hyperslab")

        return layouts

    def write(self, filename, fillvalues=None):
        """
        Write the file on disc to filename.

        Parameters
        ----------
        filename: str
          Path of the file to be written.
        fillvalues: dict, optional
          Keys are datasets names (one of: data, gain, mask) and associated
          fill value for missing data. defaults are:
            - data: nan (proc, float32) or 0 (raw, uint16)
            - gain: 0 (uint8)
            - mask: 0xffffffff (uint32)
        """
        pulse_ids = self.collect_pulse_ids()
        experiment_ids = np.char.add(np.char.add(
            self.train_ids_perframe.astype(str), ':'), pulse_ids.astype(str))

        layouts = self.collect_data()

        data_label = self.image_label
        _fillvalues = {
            # Data can be uint16 (raw) or float32 (proc)
            data_label: np.nan if layouts[data_label].dtype.kind == 'f' else 0,
            'gain': 0,
            'mask': 0xffffffff
        }
        if fillvalues:
            _fillvalues.update(fillvalues)
        # Enforce that fill values are compatible with array dtype
        _fillvalues[data_label] = layouts[data_label].dtype.type(
            _fillvalues[data_label])
        if 'gain' in layouts:
            _fillvalues['gain'] = layouts['gain'].dtype.type(
                _fillvalues['gain'])
        if 'mask' in layouts:
            _fillvalues['mask'] = layouts['mask'].dtype.type(
                _fillvalues['mask'])

        log.info("Writing to %s", filename)

        # Virtual datasets require HDF5 >= 1.10.
        # Specifying this up front should mean it fails before touching
        # the file if run on an older version. We also specify this as
        # the maximum version, to ensure we're creating files that can
        # be read by HDF5 1.10.
        with h5py.File(filename, 'w', libver=('v110', 'v110')) as f:
            f.create_dataset('cxi_version', data=[150])
            d = f.create_dataset('entry_1/experiment_identifier',
                                 shape=experiment_ids.shape,
                                 dtype=h5py.special_dtype(vlen=str))
            d[:] = experiment_ids

            # pulseId, trainId, cellId are not part of the CXI standard,
            # but it allows extra data.
            f.create_dataset(f'entry_1/{self.pulse_id_label}', data=pulse_ids)
            f.create_dataset('entry_1/trainId', data=self.train_ids_perframe)
            cellids = f.create_virtual_dataset('entry_1/cellId',
                                               layouts[self.cell_id_label])
            cellids.attrs['axes'] = 'experiment_identifier:module_identifier'

            dgrp = f.create_group('entry_1/instrument_1/detector_1')
            if len(layouts[data_label].shape) == 4:
                axes_s = 'experiment_identifier:module_identifier:y:x'
            else:
                # 5D dataset, with extra axis for
                axes_s = 'experiment_identifier:module_identifier:data_gain:y:x'
                ndg = layouts[data_label].shape[2]
                d = f.create_dataset('entry_1/data_gain', shape=(ndg,),
                                     dtype=h5py.special_dtype(vlen=str))
                d[:] = ([data_label, 'gain'] if ndg == 2 else [data_label])
                dgrp['data_gain'] = h5py.SoftLink('/entry_1/data_gain')

            data = dgrp.create_virtual_dataset(
                'data', layouts[data_label], fillvalue=_fillvalues[data_label]
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

            dgrp['experiment_identifier'] = h5py.SoftLink(
                '/entry_1/experiment_identifier')

            f['entry_1/data_1'] = h5py.SoftLink(
                '/entry_1/instrument_1/detector_1')

            dgrp.create_dataset('module_identifier', data=self.modulenos)

        log.info("Finished writing virtual CXI file")


class XtdfCXIWriter(VirtualCXIWriterBase):
    """
    Machinery to write VDS files for a group of detectors with similar
    data format - AGIPD, DSSC & LPD.

    You don't normally need to use this class directly. Instead,
    use the write_virtual_cxi() method on a multi-module detector
    data interface object.

    CXI specifies a particular layout of data in the HDF5 file format.
    It is documented here:
    http://www.cxidb.org/cxi.html

    This code writes version 1.5 CXI files.

    Parameters
    ----------
    detdata: extra_data.components.XtdfDetectorBase
      The detector data interface for the data to gather in this file.
    """
    def __init__(self, detdata) -> None:
        self.cells_per_entry = 1
        self.pulse_id_label = 'pulseId'
        self.cell_id_label = 'cellId'

        super().__init__(detdata)

    def collect_data(self):
        """
        Prepare virtual layouts and map them to the virtual sources in
        the data chunks.

        Returns
        -------
        layouts: dict
          A dictionary mapping virtual datasets names (e.g. ``data``)
          to h5py virtual layouts.
        """
        src = next(iter(self.detdata.source_to_modno))
        h5file = self.data[src].files[0].file
        image_grp = h5file['INSTRUMENT'][src][self.group_label]

        VLayout = h5py.VirtualLayout

        det_name = type(self.detdata).__name__
        if 'gain' in image_grp:
            log.info(f"Identified {det_name} calibrated data")
            shape = (self.nframes, self.nmodules) + self.detdata.module_shape
            log.info("Virtual data shape: %r", shape)

            layouts = {
                self.image_label: VLayout(
                    shape, dtype=image_grp[self.image_label].dtype),
                'gain': VLayout(shape, dtype=image_grp['gain'].dtype),
            }

            if 'mask' in image_grp:
                layouts['mask'] = VLayout(shape, dtype=image_grp['mask'].dtype)
        else:
            log.info(f"Identified {det_name} raw data")

            shape = (self.nframes, self.nmodules) + image_grp['data'].shape[1:]
            log.info("Virtual data shape: %r", shape)

            layouts = {
                self.image_label: VLayout(
                    shape, dtype=image_grp[self.image_label].dtype),
            }

        layouts[self.cell_id_label] = VLayout(
            (self.nframes, self.nmodules),
            dtype=image_grp[self.cell_id_label].dtype
        )

        return self._map_layouts(layouts)


class JUNGFRAUCXIWriter(VirtualCXIWriterBase):
    """
    Machinery to write VDS files for JUNGFRAU data in the same format
    as AGIPD/LPD virtual datasets.

    You don't normally need to use this class directly. Instead,
    use the write_virtual_cxi() method on a multi-module detector
    data interface object.

    CXI specifies a particular layout of data in the HDF5 file format.
    It is documented here:
    http://www.cxidb.org/cxi.html

    This code writes version 1.5 CXI files.

    Parameters
    ----------
    detdata: extra_data.components.JUNGFRAU
      The detector data interface for the data to gather in this file.
    """
    def __init__(self, detdata) -> None:
        # Check number of cells
        src = next(iter(detdata.source_to_modno))
        keydata = detdata.data[src, 'data.adc']
        self.cells_per_entry = keydata.entry_shape[0]
        self.pulse_id_label = 'memoryCell'
        self.cell_id_label = 'memoryCell'

        super().__init__(detdata)

        # For JUNGFRAU detectors modules are numbered from 1
        self.modulenos = list(range(1, self.nmodules + 1))

    def collect_data(self):
        """
        Prepare virtual layouts and map them to the virtual sources in
        the data chunks.

        Returns
        -------
        layouts: dict
          A dictionary mapping virtual datasets names (e.g. ``data``)
          to h5py virtual layouts.
        """
        src = next(iter(self.detdata.source_to_modno))
        h5file = self.data[src].files[0].file
        image_grp = h5file['INSTRUMENT'][src][self.group_label]

        VLayout = h5py.VirtualLayout

        det_name = type(self.detdata).__name__
        log.info(f"Identified {det_name} data")
        shape = (self.nframes, self.nmodules) + self.detdata.module_shape
        log.info("Virtual data shape: %r", shape)

        layouts = {
            self.image_label: VLayout(
                shape, dtype=image_grp[self.image_label].dtype),
            'gain': VLayout(shape, dtype=image_grp['gain'].dtype),
            self.cell_id_label: VLayout(
                (self.nframes, self.nmodules),
                dtype=image_grp[self.cell_id_label].dtype
            ),
        }

        if 'mask' in image_grp:
            layouts['mask'] = VLayout(shape, dtype=image_grp['mask'].dtype)

        return self._map_layouts(layouts)
