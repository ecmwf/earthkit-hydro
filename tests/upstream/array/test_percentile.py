import numpy as np
import pytest
from _test_inputs.readers import cama_nextxy_1, cama_nextxy_2, d8_ldd_1
from utils import (
    chain_network,
    confluence_network,
    convert_to_2d,
    make_field,
    star_network,
)

import earthkit.hydro as ekh

try:
    from earthkit.hydro import _rust  # noQA: F401

    RUST = True
except ImportError:
    RUST = False

pytestmark = pytest.mark.skipif(not RUST, reason="Rust extension required for percentiles")

NETWORKS = [("cama_nextxy", cama_nextxy_1), ("cama_nextxy", cama_nextxy_2), ("d8_ldd", d8_ldd_1)]
PERCENTILES = [0.0, 0.1, 0.25, 0.5, 0.63, 0.75, 0.9, 1.0]
FIELD = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0])
WEIGHTS = np.array([2.0, 1.0, 5.0, 1.0, 3.0, 1.0, 4.0, 1.0, 2.0, 1.0])

# Contributing areas of confluence_network(): sources {2,3} merge at 1 -> 0.
CONFLUENCE_AREAS = [[0, 1, 2, 3], [1, 2, 3], [2], [3]]


def percentile(river_network, field, p, weights=None):
    return ekh.upstream.array.percentile(river_network, field, p=p, node_weights=weights, return_type="masked")


def numpy_percentile(values, p, weights=None):
    return np.percentile(values, p * 100, method="inverted_cdf", weights=weights)


# --- correctness vs NumPy's inverted_cdf over known contributing areas ------


@pytest.mark.parametrize("p", PERCENTILES)
def test_unweighted_matches_numpy_over_a_chain(p):
    # Node k's contributing area in the chain is the slice {k, ..., n-1}.
    result = percentile(chain_network(len(FIELD)), FIELD, p)
    expected = [numpy_percentile(FIELD[k:], p) for k in range(len(FIELD))]
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("p", PERCENTILES)
def test_unweighted_matches_numpy_over_a_confluence(p):
    field = np.array([5.0, 1.0, 9.0, 3.0])
    result = percentile(confluence_network(), field, p)
    expected = [numpy_percentile(field[area], p) for area in CONFLUENCE_AREAS]
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("p", PERCENTILES)
def test_weighted_matches_numpy_over_a_chain(p):
    result = percentile(chain_network(len(FIELD)), FIELD, p, WEIGHTS)
    expected = [numpy_percentile(FIELD[k:], p, WEIGHTS[k:]) for k in range(len(FIELD))]
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("p", PERCENTILES)
def test_weighted_matches_numpy_over_a_confluence(p):
    field = np.array([5.0, 1.0, 9.0, 3.0])
    weights = np.array([2.0, 3.0, 1.0, 4.0])
    result = percentile(confluence_network(), field, p, weights)
    expected = [numpy_percentile(field[area], p, weights[area]) for area in CONFLUENCE_AREAS]
    np.testing.assert_allclose(result, expected)


# --- field patterns ---------------------------------------------------------


def test_source_nodes_return_their_own_value():
    # Leaves aggregate only themselves, so every quantile is the node's value.
    result = percentile(star_network(len(FIELD)), FIELD, 0.5)
    np.testing.assert_array_equal(result[1:], FIELD[1:])


@pytest.mark.parametrize("p", PERCENTILES)
def test_constant_field_returns_the_constant(p):
    field = np.full(len(FIELD), 7.5)
    np.testing.assert_array_equal(percentile(chain_network(len(FIELD)), field, p), field)


@pytest.mark.parametrize("p", PERCENTILES)
def test_repeated_values_match_numpy(p):
    field = np.array([2.0, 2.0, 2.0, 5.0, 5.0, 1.0, 1.0, 1.0, 9.0, 2.0])
    result = percentile(chain_network(len(field)), field, p)
    expected = [numpy_percentile(field[k:], p) for k in range(len(field))]
    np.testing.assert_allclose(result, expected)


def test_result_is_always_one_of_the_aggregated_values():
    # inverted_cdf never interpolates: the result is one of the input values.
    field = np.array([0.0, 10.0, 20.0, 30.0])
    weights = np.array([1.0, 5.0, 2.0, 1.0])
    for p in PERCENTILES:
        assert percentile(star_network(4), field, p, weights)[0] in field


# --- weighting behaviour ----------------------------------------------------


@pytest.mark.parametrize("weights, expected", [([1.0, 9.0], 10.0), ([9.0, 1.0], 0.0), ([1.0, 1.0], 0.0)])
def test_two_node_median_follows_the_weight(weights, expected):
    field = np.array([0.0, 10.0])
    result = percentile(star_network(2), field, 0.5, np.array(weights))
    assert result[0] == expected


@pytest.mark.parametrize("p", PERCENTILES)
def test_zero_weight_node_is_excluded(p):
    # A zero-weight node carries no probability mass, as if removed from the area.
    field = np.array([0.0, 10.0, 20.0, 30.0])
    weights = np.array([1.0, 0.0, 1.0, 1.0])
    remaining = np.array([0.0, 20.0, 30.0])
    result = percentile(star_network(4), field, p, weights)
    assert result[0] == numpy_percentile(remaining, p)


# --- consistency and API contract, across network topologies ----------------


@pytest.mark.parametrize("river_network", NETWORKS, indirect=True)
@pytest.mark.parametrize("p", PERCENTILES)
def test_unit_weights_match_unweighted(river_network, p):
    field = make_field(river_network)
    weights = np.ones(river_network.n_nodes)
    np.testing.assert_array_equal(
        percentile(river_network, field, p, weights),
        percentile(river_network, field, p),
    )


@pytest.mark.parametrize("river_network", NETWORKS, indirect=True)
@pytest.mark.parametrize("weighted", [False, True])
def test_endpoints_agree_with_min_and_max_operations(river_network, weighted):
    field = make_field(river_network)
    weights = np.arange(1, river_network.n_nodes + 1, dtype=float) if weighted else None
    minimum = ekh.upstream.array.min(river_network, field, return_type="masked")
    maximum = ekh.upstream.array.max(river_network, field, return_type="masked")
    np.testing.assert_allclose(percentile(river_network, field, 0.0, weights), minimum)
    np.testing.assert_allclose(percentile(river_network, field, 1.0, weights), maximum)


@pytest.mark.parametrize("river_network", NETWORKS, indirect=True)
def test_percentile_is_non_decreasing_in_p(river_network):
    field = make_field(river_network)
    weights = np.arange(1, river_network.n_nodes + 1, dtype=float)
    previous = percentile(river_network, field, 0.0, weights)
    for p in (0.2, 0.4, 0.6, 0.8, 1.0):
        current = percentile(river_network, field, p, weights)
        assert np.all(current >= previous)
        previous = current


@pytest.mark.parametrize("river_network", NETWORKS, indirect=True)
def test_gridded_result_matches_masked(river_network):
    field = make_field(river_network)
    masked = percentile(river_network, field, 0.5)
    gridded = ekh.upstream.array.percentile(
        river_network, convert_to_2d(river_network, field, 0), p=0.5, return_type="gridded"
    )
    np.testing.assert_array_equal(gridded.flatten()[river_network.mask], masked)


# --- documented error contracts ---------------------------------------------


def test_edge_weights_are_rejected():
    with pytest.raises(NotImplementedError):
        ekh.upstream.array.percentile(chain_network(4), np.arange(4.0), p=0.5, edge_weights=np.ones(3))


def test_invalid_return_type_raises():
    with pytest.raises(ValueError):
        ekh.upstream.array.percentile(chain_network(4), np.arange(4.0), p=0.5, return_type="bogus")
