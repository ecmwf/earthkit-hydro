import numpy as np
import pytest
import xarray as xr
from _test_inputs.movement import *
from _test_inputs.readers import *

import earthkit.hydro as ekh


@pytest.mark.parametrize(
    "river_network, expected",
    [
        (("cama_nextxy", cama_nextxy_1), downstream_1),
    ],
    indirect=["river_network"],
)
def test_move_upstream_xarray(river_network, expected):
    """Test move upstream with xarray input."""
    field = np.arange(1, river_network.n_nodes + 1)
    field_da = xr.DataArray(field, dims=["node_index"], coords={"node_index": np.arange(len(field))})
    result = ekh.move.upstream(river_network, field_da, return_type="masked")
    assert isinstance(result, xr.DataArray)
    np.testing.assert_allclose(result.values, expected)
