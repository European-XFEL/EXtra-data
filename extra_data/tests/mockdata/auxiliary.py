
import h5py

from .base import DeviceBase


class AuxiliarySourceBase(DeviceBase):
    def write_control(self, f):
        return  # Auxiliary sources never have CONTROL data.

    def write_instrument(self, f):
        super().write_instrument(f)

        for channel in self.output_channels:
            pipeline, _, _ = channel.partition('/')

            # Move from INSTRUMENT to the proper section, if not yet
            # done by an earlier channel.
            if f'/INSTRUMENT/{self.device_id}:{pipeline}' in f:
                f.move(f'/INSTRUMENT/{self.device_id}:{pipeline}',
                       f'/{self.section}/{self.device_id}:{pipeline}')

            # Delete trainId key automatically created by DeviceBase.
            del f[f'/{self.section}/{self.device_id}:{channel}/trainId']

            if '/' not in channel:
                # Auxiliary sources may lack the pipeline suffix, which
                # is forced by DeviceBase.
            
                f.move(f'/INDEX/{self.device_id}:{channel}',
                       f'/INDEX/{self.device_id}/{channel}')
                f.move(f'/{self.section}/{self.device_id}:{channel}',
                       f'/{self.section}/{self.device_id}/{channel}')

    def datasource_ids(self):
        for channel in self.output_channels:
            # Auxiliary sources may lack a pipeline suffix.
            sep = ':' if '/' in channel else '/'
            yield f'{self.section}/{self.device_id}{sep}{channel}'


class AgipdPulseSelection(AuxiliarySourceBase):
    section = 'REDUCTION'
    output_channels = ('xtdf/image',)
    instrument_keys = sum(
        [[(f'{key}/indicesBeforeReduction', 'u4', (3000,)),
          (f'{key}/reductionRatio', 'f8', ()),
          (f'{key}/reductionResults', 'u4', ()),
          (f'{key}/wasPassedThrough', 'u1', ()),
          (f'{key}/wasReduced', 'u1', ())]
          for key in ['cellId', 'data', 'length', 'pulseId', 'status']
        ], [])


class LitFrameFinderAux(AuxiliarySourceBase):
    section = 'REDUCTION'
    output_channels = ('daqFilter/data',)
    instrument_keys = [('dataFramePattern', 'u1', (352,))]


vlen_string_dt = h5py.string_dtype()
class TrainsOutsideBufferRange(AuxiliarySourceBase):
    section = 'ERRATA'
    output_channels = ('event',)
    instrument_keys = [
        ('deviceIds', vlen_string_dt, (11,)),
        ('originalTrainIds', 'u8', (11,)),
        ('properties', vlen_string_dt, (11,)),
        ('types', vlen_string_dt, (11,)),
        ('values', vlen_string_dt, (11,))]
