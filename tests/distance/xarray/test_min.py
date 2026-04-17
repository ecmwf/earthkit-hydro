import numpy as np
import pytest
import xarray as xr
from _test_inputs.distance import *
from _test_inputs.readers import *

import earthkit.hydro as ekh


@pytest.mark.parametrize(
    "river_network, stations_list, upstream, downstream, weights, result",
    [
        (
            ("cama_nextxy", cama_nextxy_1),
            stations,
            True,
            True,
            weights_1,
            distance_1_min_both,
        ),
    ],
    indirect=["river_network"],
)
def test_distance_min_xarray(river_network, stations_list, upstream, downstream, weights, result):
    """Test distance min with xarray input."""
    weights_da = xr.DataArray(weights, dims=["node"])
    dist = ekh.distance.min(
        river_network,
        stations_list,
        upstream=upstream,
        downstream=downstream,
        field=weights_da,
    )
    assert isinstance(dist, xr.DataArray)
    np.testing.assert_allclose(dist.values, result)
