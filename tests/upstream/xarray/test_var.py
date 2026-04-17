import numpy as np
import pytest
import xarray as xr
from _test_inputs.accumulation import *
from _test_inputs.readers import *

import earthkit.hydro as ekh


@pytest.mark.parametrize(
    "river_network, input_field, mv",
    [
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1c,
            mv_1c,
        ),
    ],
    indirect=["river_network"],
)
def test_upstream_var_xarray(river_network, input_field, mv):
    """Test upstream variance with xarray input."""
    # Convert input to xarray DataArray
    field_da = xr.DataArray(input_field, dims=["node"])

    # Call xarray-wrapped function
    result = ekh.upstream.var(river_network, field_da, return_type="masked")

    # Result should be xarray
    assert isinstance(result, xr.DataArray)

    # Variance should be non-negative
    assert np.all(result.values[~np.isnan(result.values)] >= 0)

    # Test that variance is 0 for uniform fields
    uniform_da = xr.DataArray(np.ones(river_network.n_nodes), dims=["node"])
    var_uniform = ekh.upstream.var(river_network, uniform_da, return_type="masked")
    np.testing.assert_allclose(var_uniform.values, 0, atol=1e-10)
