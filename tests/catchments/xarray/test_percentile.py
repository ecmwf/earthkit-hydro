import numpy as np
import pytest
import xarray as xr
from _test_inputs.catchment import catchment_query_field_1
from _test_inputs.readers import cama_nextxy_1
from utils import gridded_network, make_field, to_dataarray

import earthkit.hydro as ekh

try:
    from earthkit.hydro import _rust  # noQA: F401

    RUST = True
except ImportError:
    RUST = False

pytestmark = pytest.mark.skipif(not RUST, reason="Rust extension required for percentiles")

LOCATIONS = catchment_query_field_1


@pytest.mark.parametrize("p", [0.25, 0.5, 0.75])
@pytest.mark.parametrize("weighted", [False, True])
def test_xarray_wrapper_matches_array_backend(p, weighted):
    river_network = gridded_network(cama_nextxy_1)
    field = make_field(river_network)
    weights = np.arange(1, river_network.n_nodes + 1, dtype=float) if weighted else None

    result = ekh.catchments.percentile(
        river_network,
        to_dataarray(river_network, field),
        p=p,
        locations=LOCATIONS,
        node_weights=to_dataarray(river_network, weights) if weighted else None,
    )
    expected = ekh.catchments.array.percentile(river_network, field, p=p, locations=LOCATIONS, node_weights=weights)
    assert isinstance(result, xr.DataArray)
    np.testing.assert_array_equal(result.values, expected)
