import numpy as np
import pandas as pd
import pytest
import xarray as xr

from extra_data.utils import unstack_regular


def test_unstack_regular():
    stacked = xr.DataArray(
        np.arange(4*5*6*2).reshape((1, 4*5*6, 2)),
        dims=('a', 'combined', 'e'),
        coords={'combined': pd.MultiIndex.from_product(
            [range(4), range(5), range(6)], names=['b', 'c', 'd']
        ), 'e': ['x', 'y']}
    )

    res = unstack_regular(stacked, 'combined', fallback=False)
    assert res.shape == (1, 4, 5, 6, 2)
    assert res.dims == ('a', 'b', 'c', 'd', 'e')
    xr.testing.assert_equal(
        res, stacked.unstack('combined').transpose('a', 'b', 'c', 'd', 'e')
    )

    with pytest.raises(ValueError, match="cartesian"):
        # Not a cartesian product
        unstack_regular(stacked[:, :-1], 'combined', fallback=False)

    # Test the fallback to arr.unstack()
    fallback_res = unstack_regular(stacked[:, :-1], fallback=True)
    assert fallback_res.shape == (1, 4, 5, 6, 2)
    assert fallback_res.dims == ('a', 'b', 'c', 'd', 'e')
    assert fallback_res.dtype.kind == 'f'
