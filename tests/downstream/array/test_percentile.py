import numpy as np
import pytest
from _test_inputs.accumulation import *
from _test_inputs.readers import *
from utils import convert_to_2d

import earthkit.hydro as ekh


@pytest.mark.parametrize(
    "river_network, input_field, expected, p",
    [
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1c,
            downstream_metric_percentile_p05_1c,
            0.5,
        ),
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1c,
            downstream_metric_percentile_p025_1c,
            0.25,
        ),
    ],
    indirect=["river_network"],
)
def test_downstream_percentile_unweighted(river_network, input_field, expected, p):
    output = ekh.downstream.array.percentile(
        river_network, input_field, p=p, node_weights=None, return_type="masked"
    )
    np.testing.assert_allclose(output, expected)

    # Also test with 2D (gridded) input field
    input_2d = convert_to_2d(river_network, input_field, 0)
    expected_2d = convert_to_2d(river_network, expected, 0)
    output_2d = ekh.downstream.array.percentile(
        river_network, input_2d, p=p, node_weights=None
    ).flatten()
    np.testing.assert_allclose(output_2d, expected_2d.flatten())


@pytest.mark.parametrize(
    "river_network, input_field, expected",
    [
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1c,
            downstream_metric_percentile_weighted_p05_1c,
        ),
    ],
    indirect=["river_network"],
)
def test_downstream_percentile_weighted(river_network, input_field, expected):
    node_weights = np.arange(1, river_network.n_nodes + 1, dtype="float64")
    output = ekh.downstream.array.percentile(
        river_network, input_field, p=0.5, node_weights=node_weights, return_type="masked"
    )
    np.testing.assert_allclose(output, expected)


@pytest.mark.parametrize(
    "river_network, input_field, expected",
    [
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1c,
            downstream_metric_percentile_gridded_p05_1c,
        ),
    ],
    indirect=["river_network"],
)
def test_downstream_percentile_gridded_return_type(river_network, input_field, expected):
    input_2d = convert_to_2d(river_network, input_field, 0)
    output = ekh.downstream.array.percentile(
        river_network, input_2d, p=0.5, node_weights=None, return_type="gridded"
    )
    np.testing.assert_allclose(output, expected)
