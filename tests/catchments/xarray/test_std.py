import numpy as np
import pytest
import xarray as xr
from _test_inputs.catchment import *
from _test_inputs.accumulation import input_field_1c
from _test_inputs.readers import *

import earthkit.hydro as ekh


@pytest.mark.parametrize(
    "river_network, field, locations",
    [
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1c,
            catchment_query_field_1,
        ),
    ],
    indirect=["river_network"],
)
def test_catchments_std_xarray(river_network, field, locations):
    """Test catchment standard deviation with xarray input."""
    field_da = xr.DataArray(field, dims=["node_index"], coords={"node_index": np.arange(len(field))})
    result = ekh.catchments.std(river_network, field_da, locations=locations)
    assert isinstance(result, xr.DataArray)
    assert np.all(result.values >= 0)

    uniform_da = xr.DataArray(np.ones(river_network.n_nodes), dims=["node"])
    std_uniform = ekh.catchments.std(river_network, uniform_da, locations=locations)
    np.testing.assert_allclose(std_uniform.values, 0, atol=1e-10)

    var_result = ekh.catchments.var(river_network, field_da, locations=locations)
    np.testing.assert_allclose(result.values**2, var_result.values, rtol=1e-10)
