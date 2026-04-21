import numpy as np
import pytest
import xarray as xr
from _test_inputs.accumulation import input_field_1b
from _test_inputs.catchment import *
from _test_inputs.readers import *

import earthkit.hydro as ekh


@pytest.mark.parametrize(
    "river_network, field, locations, expected",
    [
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1b,
            catchment_query_field_1,
            catchment_mode_1b,
        ),
    ],
    indirect=["river_network"],
)
def test_catchments_mode_xarray(river_network, field, locations, expected):
    """Test catchment mode aggregation with xarray."""
    field_da = xr.DataArray(
        field,
        dims=["node_index"],
        coords={"node_index": np.arange(len(field))},
    )
    result = ekh.catchments.mode(river_network, field_da, locations=locations)
    assert isinstance(result, xr.DataArray)
    np.testing.assert_array_equal(result.values, expected)
