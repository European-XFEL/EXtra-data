import numpy as np
import h5py

from .base import DeviceBase

class XGM(DeviceBase):
    control_keys = [
        ('state', h5py.string_dtype(), ()),
        ('beamPosition/ixPos', 'f4', ()),
        ('beamPosition/iyPos', 'f4', ()),
        ('current/bottom/output', 'f4', ()),
        ('current/bottom/rangeCode', 'i4', ()),
        ('current/left/output', 'f4', ()),
        ('current/left/rangeCode', 'i4', ()),
        ('current/right/output', 'f4', ()),
        ('current/right/rangeCode', 'i4', ()),
        ('current/top/output', 'f4', ()),
        ('current/top/rangeCode', 'i4', ()),
        ('gasDosing/measuredPressure', 'f4', ()),
        ('gasDosing/pressureSetPoint', 'f4', ()),
        ('gasSupply/gasTypeId', 'i4', ()),
        ('gasSupply/gsdCompatId', 'i4', ()),
        ('pollingInterval', 'i4', ()),
        ('pressure/dcr', 'f4', ()),
        ('pressure/gasType', 'i4', ()),
        ('pressure/pressure1', 'f4', ()),
        ('pressure/pressureFiltered', 'f4', ()),
        ('pressure/rd', 'f4', ()),
        ('pressure/rsp', 'f4', ()),
        ('pulseEnergy/conversion', 'f8', ()),
        ('pulseEnergy/crossUsed', 'f4', ()),
        ('pulseEnergy/gammaUsed', 'f4', ()),
        ('pulseEnergy/gmdError', 'i4', ()),
        ('pulseEnergy/nummberOfBrunches', 'f4', ()),
        ('pulseEnergy/photonFlux', 'f4', ()),
        ('pulseEnergy/pressure', 'f4', ()),
        ('pulseEnergy/temperature', 'f4', ()),
        ('pulseEnergy/usedGasType', 'i4', ()),
        ('pulseEnergy/wavelengthUsed', 'f4', ()),
        ('signalAdaption/dig', 'i4', ()),
    ]

    extra_run_values = [
        ('classId', None, 'DoocsXGM'),
    ]

    # Technically, only the part before the / is the output channel.
    # But there is a structure associated with the part one level after that,
    # and we don't know what else to call it.
    output_channels = ('output/data',)

    instrument_keys = [
        ('intensityTD', 'f4', (1000,)),
        ('intensityAUXTD', 'f4', (1000,)),
        ('intensitySigma/x_data', 'f4', (1000,)),
        ('intensitySigma/y_data', 'f4', (1000,)),
        ('xTD', 'f4', (1000,)),
        ('yTD', 'f4', (1000,)),
    ]

    def write_instrument(self, f):
        super().write_instrument(f)

        # Annotate intensityTD with some units to test retrieving them
        # Karabo stores ASCII strings, assigning bytes is a shortcut to mimic that
        ds = f[f'INSTRUMENT/{self.device_id}:output/data/intensityTD']
        ds.attrs['metricPrefixEnum']= np.array([14], dtype=np.int32)
        ds.attrs['metricPrefixName'] = b'micro'
        ds.attrs['metricPrefixSymbol'] = b'u'
        ds.attrs['unitEnum'] = np.array([15], dtype=np.int32)
        ds.attrs['unitName'] = b'joule'
        ds.attrs['unitSymbol'] = b'J'

        # Also annotate a CONTROL key, where attributes are split across
        # the parent key group and the value dataset.
        # (The timestamp dataset has its own, but distinct attributes)
        # Specific examples taken from p5696, r32
        grp = f[f'CONTROL/{self.device_id}/beamPosition/ixPos']
        grp.attrs['alias'] = b'IX.POS'
        grp.attrs['description'] = b'Calculated X position [mm]'
        grp.attrs['daqPolicy'] = np.array([-1], dtype=np.int32)

        # daqPolicy is intentionally different, the correct schema value
        # is -1 as above!
        ds = grp['value']
        ds.attrs['alias'] = b'IX.POS'
        ds.attrs['daqPolicy'] = np.array([1], dtype=np.int32)

        grp = f[f'CONTROL/{self.device_id}/state']
        grp['value'][:self.ntrains] = b'ON'
        grp['value'][:5] = b'OFF'
