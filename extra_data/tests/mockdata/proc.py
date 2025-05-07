
"""Script that creates mock-data for virtual processed devices.

These are virtual devices that do not actually exist in Karabo for real,
but are the result of processing raw data into a scientifically more
useful representation or as a form of data reduction.
"""

import numpy as np

from .base import DeviceBase


class ReconstructedDLD6(DeviceBase):
    """
    Reconstructed DLD6 data from ADQ digitizer traces.
    Based on example /gpfs/exfel/exp/SQS/202101/p002448/proc/r0210/CORR-R0210-REMI01-S00000.h5
    """

    hits_dt = np.dtype([
        ('x', 'f8'), ('y', 'f8'), ('t', 'f8'), ('m', 'i4')
    ])
    signals_dt = np.dtype([
        (key, 'f8') for key in ['u1', 'u2', 'v1', 'v2', 'w1', 'w2', 'mcp']
    ])

    output_channels = ('output/rec',)
    instrument_keys = [('signals', signals_dt, (50,)),
                       ('hits', hits_dt, (50,))]

    extra_run_values = [('digitizer/baseline_region', None, ':1000', ())]
