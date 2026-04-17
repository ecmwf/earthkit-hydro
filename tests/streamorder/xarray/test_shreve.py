import numpy as np
import pytest
import xarray as xr
from _test_inputs.readers import *
from _test_inputs.streamorder import *

import earthkit.hydro as ekh


@pytest.mark.parametrize(
    "river_network, result",
    [
        (
            ("cama_nextxy", cama_nextxy_1),
            shreve_1,
        ),
    ],
    indirect=["river_network"],
)
def test_streamorder_shreve_xarray(river_network, result):
    """Test Shreve stream order with xarray output."""
    streamorder = ekh.streamorder.shreve(river_network, return_type="masked")
    assert isinstance(streamorder, xr.DataArray)
    np.testing.assert_allclose(streamorder.values, result)
