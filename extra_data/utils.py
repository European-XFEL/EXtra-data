"""
Helpers functions for the euxfel_h5tools package.

Copyright (c) 2017, European X-Ray Free-Electron Laser Facility GmbH
All rights reserved.

You should have received a copy of the 3-Clause BSD License along with this
program. If not, see <https://opensource.org/licenses/BSD-3-Clause>
"""

import os
from warnings import warn

import h5py
import numpy as np


__all__ = ['hdf5_paths', 'QuickView']


class QuickView:
    """Pun intended
        This object displays a 3D array as provided by calibrated
        hdf5 files.
        Given a 3D numpy array, it will provide you with a way to
        easily iterate over the pulses to display their respective
        images.

        First, instantiate and give it the data (a 3D numpy array):

            quick_v = QuickView(data)
            # or
            quick_v = QuickView()
            quick_v.data = data

        You can now iterate over it in three different ways:

            next(quick_v)

            quick_v.next()
            quick_v.previous()

            quick_v.pos = len(quick_v-1)

        You can also display a specific image without changing
        the position:

            quick_v.display(int)

    """

    _image = None
    _data = None
    _current_index = 0

    def __init__(self, data=None):
        if data:
            self.data = data

    @property
    def pos(self):
        return self._current_index

    @pos.setter
    def pos(self, pos):
        if self._data is not None:
            if 0 <= pos < len(self):
                self._current_index = pos
                self.show()
            else:
                err = "value should be 0 < value < " "{}".format(self._data.shape[0])
                raise ValueError(err)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        if not isinstance(data, np.ndarray) or len(data.shape) != 3:
            raise TypeError("Expected a 3D numpy array")

        self._data = data
        self._current_index = 0
        self.display()

    def next(self):
        if self._current_index < len(self):
            self._current_index += 1
            self.display()

    def prev(self):
        if self._current_index > 0:
            self._current_index -= 1
            self.display()

    def display(self, index=None):
        import matplotlib.pyplot as plot

        if index is None:
            index = self._current_index

        image_frame = self.data[index, :, :]

        if self._image is None:
            self._image = plot.imshow(image_frame)
        else:
            self._image.set_data(image_frame)

        self._image.axes.set_title("pulseId: {}".format(index))
        plot.draw()

    def __next__(self):
        self.next()

    def __len__(self):
        return self._data.shape[0]


def hdf5_paths(ds, indent=0, maxlen=100):
    """Deprecated: Visit and print name of all element in HDF5 file (from S Hauf)"""
    warn(
        "hdf5_paths is deprecated and likely to be removed. Try our h5glance "
        "package for a similar view of HDF5 files. If this is a problem, "
        "please contact da-support@xfel.eu .", stacklevel=2,
    )

    for k in list(ds.keys())[:maxlen]:
        print(" " * indent + k)
        if isinstance(ds[k], h5py.Group):
            hdf5_paths(ds[k], indent + 4, maxlen)
        else:
            print(" " * indent + k)


def available_cpu_cores():
    # This process may be restricted to a subset of the cores on the machine;
    # sched_getaffinity() tells us which on some Unix flavours (inc Linux)
    if hasattr(os, 'sched_getaffinity'):
        return len(os.sched_getaffinity(0))
    else:
        # Fallback, inc on Windows
        ncpu = os.cpu_count() or 2
        return min(ncpu, 8)
