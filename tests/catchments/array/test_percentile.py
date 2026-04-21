import numpy as np
import pytest
from _test_inputs.accumulation import *
from _test_inputs.catchment import *
from _test_inputs.readers import *

import earthkit.hydro as ekh

try:
    from earthkit.hydro import _rust  # noQA: F401

    RUST = True
except Exception:
    RUST = False


@pytest.mark.skipif(not RUST, reason="Rust unavailable")
@pytest.mark.parametrize(
    "river_network, input_field, locations, expected, p",
    [
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1c,
            catchment_query_field_1,
            catchment_percentile_p05_1c,
            0.5,
        ),
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1c,
            catchment_query_field_1,
            catchment_percentile_p025_1c,
            0.25,
        ),
    ],
    indirect=["river_network"],
)
def test_catchments_percentile_unweighted(river_network, input_field, locations, expected, p):
    output = ekh.catchments.array.percentile(river_network, input_field, p=p, locations=locations, node_weights=None)
    np.testing.assert_allclose(output, expected)


@pytest.mark.skipif(not RUST, reason="Rust unavailable")
@pytest.mark.parametrize(
    "river_network, input_field, locations, expected",
    [
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1c,
            catchment_query_field_1,
            catchment_percentile_weighted_p05_1c,
        ),
    ],
    indirect=["river_network"],
)
def test_catchments_percentile_weighted(river_network, input_field, locations, expected):
    node_weights = np.arange(1, river_network.n_nodes + 1, dtype="float64")
    output = ekh.catchments.array.percentile(
        river_network,
        input_field,
        p=0.5,
        locations=locations,
        node_weights=node_weights,
    )
    np.testing.assert_allclose(output, expected)
