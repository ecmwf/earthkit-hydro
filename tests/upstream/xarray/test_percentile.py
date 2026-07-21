import numpy as np
import pytest
import xarray as xr
from _test_inputs.readers import cama_nextxy_1
from utils import gridded_network, make_field, to_dataarray

import earthkit.hydro as ekh

try:
    from earthkit.hydro import _rust  # noQA: F401

    RUST = True
except ImportError:
    RUST = False

pytestmark = pytest.mark.skipif(not RUST, reason="Rust extension required for percentiles")


@pytest.mark.parametrize("p", [0.25, 0.5, 0.75])
@pytest.mark.parametrize("weighted", [False, True])
def test_xarray_wrapper_matches_array_backend(p, weighted):
    # The xarray wrapper must return a DataArray whose values equal the array backend.
    river_network = gridded_network(cama_nextxy_1)
    field = make_field(river_network)
    weights = np.arange(1, river_network.n_nodes + 1, dtype=float) if weighted else None

    result = ekh.upstream.percentile(
        river_network,
        to_dataarray(river_network, field),
        p=p,
        node_weights=to_dataarray(river_network, weights) if weighted else None,
        return_type="masked",
    )
    expected = ekh.upstream.array.percentile(river_network, field, p=p, node_weights=weights, return_type="masked")
    assert isinstance(result, xr.DataArray)
    np.testing.assert_array_equal(result.values, expected)


def test_xarray_gridded_return_type():
    river_network = gridded_network(cama_nextxy_1)
    field = make_field(river_network)
    result = ekh.upstream.percentile(river_network, to_dataarray(river_network, field), p=0.5, return_type="gridded")
    masked = ekh.upstream.array.percentile(river_network, field, p=0.5, return_type="masked")
    assert isinstance(result, xr.DataArray)
    np.testing.assert_array_equal(result.values.flatten()[river_network.mask], masked)
