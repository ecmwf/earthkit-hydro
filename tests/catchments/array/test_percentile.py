import numpy as np
import pytest
from _test_inputs.accumulation import input_field_1c
from _test_inputs.catchment import catchment_query_field_1
from _test_inputs.readers import cama_nextxy_1

import earthkit.hydro as ekh

try:
    from earthkit.hydro import _rust  # noQA: F401

    RUST = True
except ImportError:
    RUST = False

pytestmark = pytest.mark.skipif(not RUST, reason="Rust extension required for percentiles")

NETWORK = [("cama_nextxy", cama_nextxy_1)]
LOCATIONS = catchment_query_field_1  # node indices of the catchment outlets
PERCENTILES = [0.0, 0.25, 0.5, 0.75, 1.0]


@pytest.mark.parametrize("river_network", NETWORK, indirect=True)
@pytest.mark.parametrize("p", PERCENTILES)
@pytest.mark.parametrize("weighted", [False, True])
def test_catchment_percentile_is_upstream_percentile_at_the_outlet(river_network, p, weighted):
    # A catchment is the contributing area of its outlet, so the catchment
    # percentile must equal the upstream percentile evaluated at that outlet node.
    field = np.asarray(input_field_1c, dtype=float)
    weights = np.arange(1, river_network.n_nodes + 1, dtype=float) if weighted else None

    catchment = np.asarray(
        ekh.catchments.array.percentile(river_network, field, p=p, locations=LOCATIONS, node_weights=weights)
    )
    upstream = np.asarray(
        ekh.upstream.array.percentile(river_network, field, p=p, node_weights=weights, return_type="masked")
    )
    np.testing.assert_allclose(catchment, upstream[np.asarray(LOCATIONS)])


@pytest.mark.parametrize("river_network", NETWORK, indirect=True)
@pytest.mark.parametrize("weight", [1.0, 4.2])
@pytest.mark.parametrize("p", PERCENTILES)
def test_uniform_weights_match_unweighted(river_network, weight, p):
    field = np.asarray(input_field_1c, dtype=float)
    weights = np.full(river_network.n_nodes, weight)
    weighted = ekh.catchments.array.percentile(river_network, field, p=p, locations=LOCATIONS, node_weights=weights)
    unweighted = ekh.catchments.array.percentile(river_network, field, p=p, locations=LOCATIONS, node_weights=None)
    np.testing.assert_allclose(np.asarray(weighted), np.asarray(unweighted))
