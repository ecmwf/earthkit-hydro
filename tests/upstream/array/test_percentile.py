import numpy as np
import pytest
from _test_inputs.accumulation import input_field_1c
from _test_inputs.readers import cama_nextxy_1
from utils import chain_network, convert_to_2d, star_network

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


def upstream_percentile(river_network, field, p, weights=None):
    return np.asarray(
        ekh.upstream.array.percentile(river_network, field, p=p, node_weights=weights, return_type="masked")
    )


# --------------------------------------------------------------------------- #
# Unweighted: equals NumPy's percentile over each node's contributing area     #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("p", PERCENTILES)
def test_unweighted_matches_numpy_over_contributing_area(p):
    # In the chain 0 <- 1 <- ... <- 9, node k's contributing area is {k, ..., 9},
    # so its percentile is NumPy's percentile of that slice of the field.
    output = upstream_percentile(chain_network(len(FIELD)), FIELD, p)
    expected = [np.percentile(FIELD[k:], p * 100) for k in range(len(FIELD))]
    np.testing.assert_allclose(output, expected)


@pytest.mark.parametrize("river_network", NETWORK, indirect=True)
def test_endpoints_agree_with_min_and_max_operations(river_network):
    # p=0 and p=1 must coincide with the dedicated upstream min / max operations.
    field = input_field_1c
    np.testing.assert_allclose(
        upstream_percentile(river_network, field, 0.0),
        ekh.upstream.array.min(river_network, field, return_type="masked"),
    )
    np.testing.assert_allclose(
        upstream_percentile(river_network, field, 1.0),
        ekh.upstream.array.max(river_network, field, return_type="masked"),
    )


@pytest.mark.parametrize("river_network", NETWORK, indirect=True)
def test_percentile_is_non_decreasing_in_p(river_network):
    field = input_field_1c
    weights = np.arange(1, river_network.n_nodes + 1, dtype=float)
    previous = upstream_percentile(river_network, field, 0.0, weights)
    for p in (0.2, 0.4, 0.6, 0.8, 1.0):
        current = upstream_percentile(river_network, field, p, weights)
        assert np.all(current >= previous - 1e-9)
        previous = current


@pytest.mark.parametrize("river_network", NETWORK, indirect=True)
def test_gridded_result_matches_masked(river_network):
    field = input_field_1c
    masked = upstream_percentile(river_network, field, 0.5)
    field_2d = convert_to_2d(river_network, field, 0)
    gridded = ekh.upstream.array.percentile(river_network, field_2d, p=0.5, return_type="gridded")
    np.testing.assert_allclose(np.asarray(gridded).flatten()[river_network.mask], masked)


# --------------------------------------------------------------------------- #
# Weighted behaviour                                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("river_network", NETWORK, indirect=True)
@pytest.mark.parametrize("weight", [1.0, 4.2])
@pytest.mark.parametrize("p", PERCENTILES)
def test_uniform_weights_match_unweighted(river_network, weight, p):
    # Any constant weight must reproduce the unweighted result (scale invariant).
    field = np.asarray(input_field_1c, dtype=float)
    weights = np.full(river_network.n_nodes, weight)
    np.testing.assert_allclose(
        upstream_percentile(river_network, field, p, weights),
        upstream_percentile(river_network, field, p),
    )


def test_weights_redistribute_percentile_space():
    # Whole star drains into node 0. Values 0,10,20,30 with a heavy weight on 10:
    # edge lengths L=(w_i+w_{i+1})/2 = [5.5, 5.5, 1], S=12, so the first knot is at
    # 5.5/12 and p=0.25 interpolates within [0, 10], pulled up from the unweighted 7.5.
    field = np.array([0.0, 10.0, 20.0, 30.0])
    weights = np.array([1.0, 10.0, 1.0, 1.0])
    out = upstream_percentile(star_network(4), field, 0.25, weights)
    assert out[0] == pytest.approx(0.25 / (5.5 / 12.0) * 10.0)


def test_dominant_weight_moves_low_quantile_off_that_node():
    # A dominant weight on the smallest value stretches its intervals, so the 25th
    # percentile is no longer pinned to that value.
    field = np.array([0.0, 10.0, 20.0, 30.0])
    weights = np.array([1000.0, 1.0, 1.0, 1.0])
    out = upstream_percentile(star_network(4), field, 0.25, weights)
    assert out[0] == pytest.approx(0.25 / (500.5 / 502.5) * 10.0)
    assert out[0] > field[0]


@pytest.mark.parametrize("weights", [[1.0, 9.0], [9.0, 1.0], [1.0, 1.0]])
def test_two_node_area_is_weight_independent_midpoint(weights):
    # With only two values there is a single interval, so weights cannot redistribute
    # space and the median is the midpoint regardless of them.
    field = np.array([0.0, 10.0])
    out = upstream_percentile(star_network(2), field, 0.5, np.array(weights))
    assert out[0] == pytest.approx(5.0)
