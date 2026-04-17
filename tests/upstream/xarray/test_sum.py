import numpy as np
import pytest
import xarray as xr
from _test_inputs.accumulation import *
from _test_inputs.readers import *

import earthkit.hydro as ekh


@pytest.mark.parametrize(
    "river_network, input_field, expected, mv",
    [
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1c,
            upstream_metric_sum_1c,
            mv_1c,
        ),
    ],
    indirect=["river_network"],
)
def test_upstream_sum_xarray(river_network, input_field, expected, mv):
    """Test upstream sum with xarray input."""
    # Convert input to xarray DataArray
    field_da = xr.DataArray(input_field, dims=["node"])

    # Call xarray-wrapped function
    result = ekh.upstream.sum(river_network, field_da, return_type="masked")

    # Result should be xarray
    assert isinstance(result, xr.DataArray)

    # Check values
    np.testing.assert_allclose(result.values, expected, rtol=1e-6, equal_nan=True)
