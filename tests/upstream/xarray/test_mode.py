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
            input_field_1b,
            upstream_metric_mode_1b,
            mv_1b,
        ),
    ],
    indirect=["river_network"],
)
def test_upstream_mode_xarray(river_network, input_field, expected, mv):
    """Test upstream mode with xarray input."""
    # Convert input to xarray DataArray
    field_da = xr.DataArray(
        input_field,
        dims=["node_index"],
        coords={"node_index": np.arange(len(input_field))},
    )

    # Call xarray-wrapped function
    result = ekh.upstream.mode(river_network, field_da, return_type="masked")

    # Result should be xarray
    assert isinstance(result, xr.DataArray)

    # Check values - mode uses exact integer comparison
    np.testing.assert_array_equal(result.values, expected)
