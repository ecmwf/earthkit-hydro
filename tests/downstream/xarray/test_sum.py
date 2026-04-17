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
            downstream_metric_sum_1c,
            mv_1c,
        ),
    ],
    indirect=["river_network"],
)
def test_downstream_sum_xarray(river_network, input_field, expected, mv):
    """Test downstream sum with xarray input."""
    field_da = xr.DataArray(input_field, dims=["node_index"], coords={"node_index": np.arange(len(input_field))})
    result = ekh.downstream.sum(river_network, field_da, return_type="masked")
    assert isinstance(result, xr.DataArray)
    np.testing.assert_allclose(result.values, expected, rtol=1e-6, equal_nan=True)
