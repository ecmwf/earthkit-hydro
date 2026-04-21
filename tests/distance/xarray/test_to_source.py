import numpy as np
import pytest
import xarray as xr
from _test_inputs.distance import *
from _test_inputs.readers import *

import earthkit.hydro as ekh


@pytest.mark.parametrize(
    "river_network, field, expected",
    [
        (
            ("cama_nextxy", cama_nextxy_1),
            None,
            distance_1_to_source_shortest,
        ),
    ],
    indirect=["river_network"],
)
def test_distance_to_source_xarray(river_network, field, expected):
    """Test distance to source with xarray output."""
    result = ekh.distance.to_source(river_network, field=field, path="shortest", return_type="masked")
    assert isinstance(result, xr.DataArray)
    np.testing.assert_allclose(result.values, expected, rtol=1e-6)
