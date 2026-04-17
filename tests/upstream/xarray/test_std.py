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
def test_upstream_std_xarray(river_network, input_field, mv):
    """Test upstream standard deviation with xarray input."""
    # Convert input to xarray DataArray
    field_da = xr.DataArray(input_field, dims=["node_index"], coords={"node_index": np.arange(len(input_field))})

    # Call xarray-wrapped function
    result = ekh.upstream.std(river_network, field_da, return_type="masked")

    # Result should be xarray
    assert isinstance(result, xr.DataArray)

    # Std should be non-negative
    assert np.all(result.values[~np.isnan(result.values)] >= 0)

    # Test that std is 0 for uniform fields
    uniform_da = xr.DataArray(np.ones(river_network.n_nodes), dims=["node_index"], coords={"node_index": np.arange(river_network.n_nodes)})
    std_uniform = ekh.upstream.std(river_network, uniform_da, return_type="masked")
    np.testing.assert_allclose(std_uniform.values, 0, atol=1e-10)

    # Test relationship with variance
    var_result = ekh.upstream.var(river_network, field_da, return_type="masked")
    np.testing.assert_allclose(result.values**2, var_result.values, rtol=1e-10, equal_nan=True)
