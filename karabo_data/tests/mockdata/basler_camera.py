"""Script that creates a mock-run for the basler camera"""
from os.path import join as op_join

from .base import DeviceBase

class BaslerCamera(DeviceBase):
    """
    Basler Camera device
    Based on example /gpfs/exfel/exp/SPB/201930/p900061/raw/r0055/RAW-R0055-DA01-S00000.h5
    """

    def __init__(self, device_id, nsamples=None, px_size=None):
        """Create a dummy basler device that inherits from Device Base"""
        self.px_size = px_size or (2058, 2456)
        super(BaslerCamera, self).__init__(device_id, nsamples=nsamples)
        self.output_channels = ('daqQutput/data',)
        self.instrument_keys = [
            ('image/bitsPerPixel', 'i4', ()),
            ('image/dimTypes', 'i4', (2,)),
            ('image/dims', 'u8', (2,)),
            ('image/encoding', 'i4', ()),
            ('image/pixels', 'u2', self.px_size),
            ('image/roiOffsets', 'u8', (2,)),
            ('image/binning', 'u8', (2,)),
            ('image/flipX', 'u1', ()),
            ('image/flipY', 'u1', ())
        ]

    def write_instrument(self, f):
        super().write_instrument(f)

        # Add fixed metadata
        for channel in self.output_channels:
            image_grp = 'INSTRUMENT/{}:{}/image/'.format(self.device_id, channel)
            f[op_join(image_grp, 'bitsPerPixel')][:self.nsamples] = 16
            f[op_join(image_grp, 'dims')][:self.nsamples] = self.px_size


