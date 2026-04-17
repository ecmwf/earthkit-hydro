import numpy as np
import pytest
import xarray as xr
from _test_inputs.catchment import *
from _test_inputs.readers import *

import earthkit.hydro as ekh


@pytest.mark.parametrize(
    "river_network, locations, expected",
    [
        (
            ("cama_nextxy", cama_nextxy_1),
            catchment_query_field_1,
            catchment_1,
        ),
    ],
    indirect=["river_network"],
)
def test_catchments_find_xarray(river_network, locations, expected):
    """Test catchment find with xarray input."""
    # For find, locations is the input, not field
    result = ekh.catchments.find(river_network, locations=locations)
    assert isinstance(result, xr.DataArray)
    # Basic check that result has right shape
    assert len(result.values) == river_network.n_nodes
