import numpy as np
import re

__all__ = [
    'stack_data',
    'stack_detector_data',
]

def stack_data(train, data, axis=-3, xcept=()):
    """Stack data from devices in a train.

    For detector data, use stack_detector_data instead: it can handle missing
    modules, which this function cannot.

    The returned array will have an extra dimension. The data will be ordered
    according to any groups of digits in the source name, interpreted as
    integers. Other characters do not affect sorting. So:

        "B_7_0" < "A_12_0" < "A_12_1"

    Parameters
    ----------
    train: dict
        Train data.
    data: str
        The path to the device parameter of the data you want to stack.
    axis: int, optional
        Array axis on which you wish to stack.
    xcept: list
        List of devices to ignore (useful if you have reccored slow data with
        detector data in the same run).

    Returns
    -------
    combined: numpy.array
        Stacked data for requested data path.
    """
    devices = [dev for dev in train.keys() if dev not in xcept]

    if not devices:
        raise ValueError("No data after filtering by 'xcept' argument.")

    dtypes = set()
    ordered_arrays = []
    for device in sorted(devices, key=lambda d: list(map(int, re.findall(r'\d+', d)))):
        array = train[device][data]
        dtypes.add(array.dtype)
        ordered_arrays.append(array)

    if len(dtypes) > 1:
        raise ValueError("Arrays have mismatched dtypes: {}".format(dtypes))

    return np.stack(ordered_arrays, axis=axis)


def stack_detector_data(
        train, data, axis=-3, modules=16, fillvalue=None, real_array=True, *,
        pattern=r'/DET/(\d+)CH', starts_at=0,
):
    """Stack data from detector modules in a train.

    Parameters
    ----------
    train: dict
        Train data.
    data: str
        The path to the device parameter of the data you want to stack, e.g. 'image.data'.
    axis: int
        Array axis on which you wish to stack (default is -3).
    modules: int
        Number of modules composing a detector (default is 16).
    fillvalue: number
        Value to use in place of data for missing modules. The default is nan
        (not a number) for floating-point data, and 0 for integers.
    real_array: bool
        If True (default), copy the data together into a real numpy array.
        If False, avoid copying the data and return a limited array-like wrapper
        around the existing arrays. This is sufficient for assembling images
        using detector geometry, and allows better performance.
    pattern: str
        Regex to find the module number in source names. Should contain a group
        which can be converted to an integer. E.g. ``r'/DET/JNGFR(\\d+)'`` for
        one JUNGFRAU naming convention.
    starts_at: int
        By default, uses module numbers starting at 0 (e.g. 0-15 inclusive).
        If the numbering is e.g. 1-16 instead, pass starts_at=1. This is not
        automatic because the first or last module may be missing from the data.

    Returns
    -------
    combined: numpy.array
        Stacked data for requested data path.
    """

    if not train:
        raise ValueError("No data")

    dtypes, shapes, empty_mods = set(), set(), set()
    modno_arrays = {}
    for src in train:
        det_mod_match = re.search(pattern, src)
        if not det_mod_match:
            raise ValueError(f"Source {src!r} doesn't match pattern {pattern!r}")
        modno = int(det_mod_match.group(1)) - starts_at

        try:
            array = train[src][data]
        except KeyError:
            continue
        dtypes.add(array.dtype)
        shapes.add(array.shape)
        modno_arrays[modno] = array

    if len(dtypes) > 1:
        raise ValueError("Arrays have mismatched dtypes: {}".format(dtypes))
    if len(shapes) > 1:
        s1, s2, *_ = sorted(shapes)
        if len(shapes) > 2 or (s1[0] != 0) or (s1[1:] != s2[1:]):
            raise ValueError("Arrays have mismatched shapes: {}".format(shapes))
        empty_mods = {n for n, a in modno_arrays.items() if a.shape == s1}
        for modno in empty_mods:
            del modno_arrays[modno]
        shapes.remove(s1)
    if max(modno_arrays) >= modules:
        raise IndexError("Module {} is out of range for a detector with {} modules"
                         .format(max(modno_arrays), modules))

    dtype = dtypes.pop()
    shape = shapes.pop()

    if fillvalue is None:
        fillvalue = np.nan if dtype.kind == 'f' else 0
    fillvalue = dtype.type(fillvalue)  # check value compatibility with dtype

    stack = StackView(
        modno_arrays, modules, shape, dtype, fillvalue, stack_axis=axis
    )
    if real_array:
        return stack.asarray()

    return stack


class StackView:
    """Limited array-like object holding detector data from several modules.

    Access is limited to either a single module at a time or all modules
    together, but this is enough to assemble detector images.
    """
    def __init__(self, data, nmodules, mod_shape, dtype, fillvalue,
                 stack_axis=-3):
        self._nmodules = nmodules
        self._data = data  # {modno: array}
        self.dtype = dtype
        self._fillvalue = fillvalue
        self._mod_shape = mod_shape
        self.ndim = len(mod_shape) + 1
        self._stack_axis = stack_axis
        if self._stack_axis < 0:
            self._stack_axis += self.ndim
        sax = self._stack_axis
        self.shape = mod_shape[:sax] + (nmodules,) + mod_shape[sax:]

    def __repr__(self):
        return "<VirtualStack (shape={}, {}/{} modules, dtype={})>".format(
            self.shape, len(self._data), self._nmodules, self.dtype,
        )

    # Multidimensional slicing
    def __getitem__(self, slices):
        if not isinstance(slices, tuple):
            slices = (slices,)

        missing_dims = self.ndim - len(slices)
        if Ellipsis in slices:
            ix = slices.index(Ellipsis)
            missing_dims += 1
            slices = slices[:ix] + (slice(None, None),) * missing_dims + slices[ix + 1:]
        else:
            slices = slices + (slice(None, None),) * missing_dims

        modno = slices[self._stack_axis]
        mod_slices = slices[:self._stack_axis] + slices[self._stack_axis + 1:]

        if isinstance(modno, int):
            if modno < 0:
                modno += self._nmodules
            return self._get_single_mod(modno, mod_slices)
        elif modno == slice(None, None):
            return self._get_all_mods(mod_slices)
        else:
            raise Exception(
                "VirtualStack can only slice a single module or all modules"
            )

    def _get_single_mod(self, modno, mod_slices):
        try:
            mod_data = self._data[modno]
        except KeyError:
            if modno >= self._nmodules:
                raise IndexError(modno)
            mod_data = np.full(self._mod_shape, self._fillvalue, self.dtype)
            self._data[modno] = mod_data

        # Now slice the module data as requested
        return mod_data[mod_slices]

    def _get_all_mods(self, mod_slices):
        new_data = {modno: self._get_single_mod(modno, mod_slices)
                    for modno in self._data}
        new_mod_shape = list(new_data.values())[0].shape
        return StackView(new_data, self._nmodules, new_mod_shape, self.dtype,
                         self._fillvalue)

    def asarray(self):
        """Copy this data into a real numpy array

        Don't do this until necessary - the point of using VirtualStack is to
        avoid copying the data unnecessarily.
        """
        start_shape = (self._nmodules,) + self._mod_shape
        arr = np.full(start_shape, self._fillvalue, dtype=self.dtype)
        for modno, data in self._data.items():
            arr[modno] = data
        return np.moveaxis(arr, 0, self._stack_axis)

    def squeeze(self, axis=None):
        """Drop axes of length 1 - see numpy.squeeze()"""
        if axis is None:
            slices = [0 if d == 1 else slice(None, None) for d in self.shape]
        elif isinstance(axis, (int, tuple)):
            if isinstance(axis, int):
                axis = (axis,)

            slices = [slice(None, None)] * self.ndim

            for ax in axis:
                try:
                    slices[ax] = 0
                except IndexError:
                    raise np.AxisError(
                        "axis {} is out of bounds for array of dimension {}"
                        .format(ax, self.ndim)
                    )
                if self.shape[ax] != 1:
                    raise ValueError("cannot squeeze out an axis with size != 1")
        else:
            raise TypeError("axis={!r} not supported".format(axis))

        return self[tuple(slices)]
