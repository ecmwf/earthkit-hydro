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
def test_downstream_var_xarray(river_network, input_field, mv):
    """Test downstream variance with xarray input."""
    field_da = xr.DataArray(input_field, dims=["node"])
    result = ekh.downstream.var(river_network, field_da, return_type="masked")
    assert isinstance(result, xr.DataArray)
    assert np.all(result.values[~np.isnan(result.values)] >= 0)

    uniform_da = xr.DataArray(np.ones(river_network.n_nodes), dims=["node"])
    var_uniform = ekh.downstream.var(river_network, uniform_da, return_type="masked")
    np.testing.assert_allclose(var_uniform.values, 0, atol=1e-10)
