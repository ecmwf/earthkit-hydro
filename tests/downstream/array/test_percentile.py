import numpy as np
import pytest
from _test_inputs.accumulation import input_field_1c
from _test_inputs.readers import cama_nextxy_1
from utils import chain_network, star_network

import earthkit.hydro as ekh

try:
    from earthkit.hydro import _rust  # noQA: F401

    RUST = True
except ImportError:
    RUST = False

pytestmark = pytest.mark.skipif(not RUST, reason="Rust extension required for percentiles")

NETWORK = [("cama_nextxy", cama_nextxy_1)]
PERCENTILES = [0.0, 0.1, 0.25, 0.5, 0.63, 0.75, 0.9, 1.0]
FIELD = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0])


def downstream_percentile(river_network, field, p, weights=None):
    return np.asarray(
        ekh.downstream.array.percentile(river_network, field, p=p, node_weights=weights, return_type="masked")
    )


@pytest.mark.parametrize("p", PERCENTILES)
def test_unweighted_matches_numpy_over_draining_area(p):
    # In the chain 0 <- 1 <- ... <- 9, node k's draining area is {0, ..., k}, so its
    # percentile is NumPy's percentile of that slice of the field.
    output = downstream_percentile(chain_network(len(FIELD)), FIELD, p)
    expected = [np.percentile(FIELD[: k + 1], p * 100) for k in range(len(FIELD))]
    np.testing.assert_allclose(output, expected)


@pytest.mark.parametrize("river_network", NETWORK, indirect=True)
def test_endpoints_agree_with_min_and_max_operations(river_network):
    field = np.asarray(input_field_1c, dtype=float)
    np.testing.assert_allclose(
        downstream_percentile(river_network, field, 0.0),
        np.asarray(ekh.downstream.array.min(river_network, field, return_type="masked")),
    )
    np.testing.assert_allclose(
        downstream_percentile(river_network, field, 1.0),
        np.asarray(ekh.downstream.array.max(river_network, field, return_type="masked")),
    )


@pytest.mark.parametrize("river_network", NETWORK, indirect=True)
@pytest.mark.parametrize("weight", [1.0, 4.2])
@pytest.mark.parametrize("p", PERCENTILES)
def test_uniform_weights_match_unweighted(river_network, weight, p):
    field = np.asarray(input_field_1c, dtype=float)
    weights = np.full(river_network.n_nodes, weight)
    np.testing.assert_allclose(
        downstream_percentile(river_network, field, p, weights),
        downstream_percentile(river_network, field, p),
    )


def test_weights_redistribute_percentile_space():
    # Node 3's draining area is the whole chain {0,1,2,3} with values 0,10,20,30.
    # A heavy weight on 10 gives edge lengths [5.5, 5.5, 1] (S=12), so p=0.25 lands
    # within [0, 10] at 0.25 / (5.5/12) * 10, up from the unweighted 7.5.
    field = np.array([0.0, 10.0, 20.0, 30.0])
    weights = np.array([1.0, 10.0, 1.0, 1.0])
    out = downstream_percentile(chain_network(4), field, 0.25, weights)
    assert out[3] == pytest.approx(0.25 / (5.5 / 12.0) * 10.0)


@pytest.mark.parametrize("weights", [[1.0, 9.0], [9.0, 1.0], [1.0, 1.0]])
def test_two_node_area_is_weight_independent_midpoint(weights):
    # Node 1 drains to node 0, so its draining area is exactly {0, 1}.
    field = np.array([0.0, 10.0])
    out = downstream_percentile(star_network(2), field, 0.5, np.array(weights))
    assert out[1] == pytest.approx(5.0)
