import numpy as np
import pytest
import xarray as xr
from _test_inputs.catchment import *
from _test_inputs.accumulation import input_field_1c
from _test_inputs.readers import *

import earthkit.hydro as ekh


@pytest.mark.parametrize(
    "river_network, field, locations, expected",
    [
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1c,
            catchment_query_field_1,
            catchment_max_1c,
        ),
    ],
    indirect=["river_network"],
)
def test_catchments_max_xarray(river_network, field, locations, expected):
    """Test catchment max with xarray input."""
    field_da = xr.DataArray(field, dims=["node_index"], coords={"node_index": np.arange(len(field))})
    result = ekh.catchments.max(river_network, field_da, locations=locations)
    assert isinstance(result, xr.DataArray)
    np.testing.assert_allclose(result.values, expected, rtol=1e-6)
