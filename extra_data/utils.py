"""
Helpers functions for the euxfel_h5tools package.

Copyright (c) 2017, European X-Ray Free-Electron Laser Facility GmbH
All rights reserved.

You should have received a copy of the 3-Clause BSD License along with this
program. If not, see <https://opensource.org/licenses/BSD-3-Clause>
"""

import os
import sys
from shutil import get_terminal_size
from typing import TYPE_CHECKING

import numpy as np
if TYPE_CHECKING:
    import pandas


def available_cpu_cores():
    # This process may be restricted to a subset of the cores on the machine;
    # sched_getaffinity() tells us which on some Unix flavours (inc Linux)
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    else:
        # Fallback, inc on Windows
        ncpu = os.cpu_count() or 2
        return min(ncpu, 8)


def default_num_threads(fixed_limit=16):
    # Default to 16, EXTRA_NUM_THREADS, or available CPU cores (picking lowest)
    threads_limits = ([fixed_limit, available_cpu_cores()])
    try:
        threads_limits.append(int(os.environ['EXTRA_NUM_THREADS']))
    except (KeyError, ValueError):  # Not set, or not an integer
        pass
    return min(threads_limits)


def progress_bar(done, total, suffix=" "):
    line = f"Progress: {done}/{total}{suffix}[{{}}]"
    length = min(get_terminal_size().columns - len(line), 50)
    filled = int(length * done // total)
    bar = "#" * filled + " " * (length - filled)
    return line.format(bar)


def isinstance_no_import(obj, mod: str, cls: str):
    """Check if isinstance(obj, mod.cls) without loading mod"""
    m = sys.modules.get(mod)
    if m is None:
        return False

    return isinstance(obj, getattr(m, cls))


def _multiindex_regular_labels(mix: "pandas.MultiIndex"):
    """Return a tuple of indexes if mix is a cartesian product, else None"""
    import pandas as pd
    if mix.has_duplicates:
        return None

    k1_sel, k1_subix = mix.get_loc_level(mix[0][0])
    rpt_len = len(k1_subix)
    rpt, rem = divmod(len(mix), rpt_len)
    if rem != 0:
        return None

    if isinstance(k1_subix, pd.MultiIndex):
        inner_labels = _multiindex_regular_labels(k1_subix)
        if inner_labels is None:
            return None
    else:
        inner_labels = (k1_subix,)

    # Check that the outermost level has each value rpt_len times
    outer_codes = mix.codes[0].reshape(-1, rpt_len)
    if (outer_codes != outer_codes[:, 0, np.newaxis]).any():
        return None

    # Check that each inner level is repeating regularly
    for level in range(1, mix.nlevels):
        codes = mix.codes[level].reshape(-1, rpt_len)
        if (codes != codes[0]).any():
            return None

    outer_labels = mix.levels[0][outer_codes[:, 0]]
    return (outer_labels,) + inner_labels


def _unstack_regular_once(arr, dim, fallback=False, fill_value=None):
    import pandas as pd
    import xarray as xr

    mix = arr.indexes[dim]
    assert isinstance(mix, pd.MultiIndex)
    if (mix_labels := _multiindex_regular_labels(mix)) is None:
        # Not a cartesian product -> cannot reshape
        if fallback:
            return _unstack_fallback(arr, dim, fill_value)
        raise ValueError(f"MultiIndex for {dim!r} is not a cartesian product")

    mix_shape = tuple(len(l) for l in mix_labels)
    dim_ix = arr.dims.index(dim)
    new_shape = arr.shape[:dim_ix] + mix_shape + arr.shape[dim_ix + 1:]
    data = arr.values.reshape(new_shape)

    coords = {
        k: v for (k, v) in arr.coords.items() if dim not in v.dims  # Unchanged
    } | dict(
        zip(mix.names, mix_labels)  # Unstacked coordinates
    ) | {
        # Other coordinates along unstacked dimension
        k: (mix.names, v.values.reshape(mix_shape)) for (k, v) in arr.coords.items()
        if (dim in v.dims and k != dim and k not in mix.names)
    }

    return xr.DataArray(
        data,
        dims=arr.dims[:dim_ix] + mix.names + arr.dims[dim_ix + 1:],
        coords=coords,
    )


def _unstack_fallback(arr, dim, fill_value=None):
    from xarray.core.dtypes import NA
    if fill_value is None:
        fill_value = NA

    res = arr.unstack(dim, fill_value=fill_value)

    # Restore the obvious axis order
    dim_ix = arr.dims.index(dim)
    new_dims = res.dims[arr.ndim - 1:]
    dim_order = arr.dims[:dim_ix] + new_dims + arr.dims[dim_ix + 1:]
    return res.transpose(*dim_order)


def unstack_regular(arr, dim=None, *, fallback=False, fill_value=None):
    """Unstack an xarray.DataArray efficiently when no fill values are needed.

    Where the stacked index is a full cartesian product, we can make a view of
    the original data instead of copying it, which is much more efficient. In
    this case, we also don't have to convert integers to floats to allow for
    NaN values.

    If ``fallback=True``, this also accepts arrays where fill values are needed,
    and uses xarray's implementation. Otherwise, it raises ValueError if
    unstacking would require inserting fill values.

    The unstacked dimensions are expanded in-place in the dimension order,
    rather than being moved to the end.
    """
    import pandas as pd

    if dim is None:
        dim = [d for d in arr.dims if isinstance(arr.indexes.get(d), pd.MultiIndex)]
    if isinstance(dim, str):
        dim = [dim]
    for d in dim:
        arr = _unstack_regular_once(arr, d, fallback, fill_value)

    return arr
