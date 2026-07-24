# SPDX-FileCopyrightText: 2026- European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from _test_inputs.readers import cama_nextxy_1, cama_nextxy_2, d8_ldd_1
from utils import chain_network, make_field

import earthkit.hydro as ekh

try:
    from earthkit.hydro import _rust  # noQA: F401

    RUST = True
except ImportError:
    RUST = False

pytestmark = pytest.mark.skipif(not RUST, reason="Rust extension required for percentiles")

NETWORKS = [("cama_nextxy", cama_nextxy_1), ("cama_nextxy", cama_nextxy_2), ("d8_ldd", d8_ldd_1)]
PERCENTILES = [0.0, 0.25, 0.5, 0.75, 1.0]


def outlets(river_network):
    return [0, river_network.n_nodes // 2, river_network.n_nodes - 1]


@pytest.mark.parametrize("river_network", NETWORKS, indirect=True)
@pytest.mark.parametrize("p", PERCENTILES)
@pytest.mark.parametrize("weighted", [False, True])
def test_catchment_percentile_is_upstream_percentile_at_the_outlet(river_network, p, weighted):
    # A catchment is the contributing area of its outlet, so the catchment percentile
    # must equal the upstream percentile evaluated at that outlet node.
    field = make_field(river_network)
    locations = outlets(river_network)
    weights = np.arange(1, river_network.n_nodes + 1, dtype=float) if weighted else None

    catchment = ekh.catchments.array.percentile(river_network, field, p=p, locations=locations, node_weights=weights)
    upstream = ekh.upstream.array.percentile(river_network, field, p=p, node_weights=weights, return_type="masked")
    np.testing.assert_allclose(catchment, upstream[locations])


@pytest.mark.parametrize("p", PERCENTILES)
def test_catchment_over_a_chain_matches_numpy(p):
    # The catchment of outlet k in the chain is the slice {k, ..., n-1}.
    field = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0])
    locations = [0, 4, 9]
    result = ekh.catchments.array.percentile(chain_network(len(field)), field, p=p, locations=locations)
    expected = [np.percentile(field[k:], p * 100, method="inverted_cdf") for k in locations]
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("river_network", NETWORKS, indirect=True)
@pytest.mark.parametrize("p", PERCENTILES)
def test_unit_weights_match_unweighted(river_network, p):
    field = make_field(river_network)
    locations = outlets(river_network)
    weights = np.ones(river_network.n_nodes)
    weighted = ekh.catchments.array.percentile(river_network, field, p=p, locations=locations, node_weights=weights)
    unweighted = ekh.catchments.array.percentile(river_network, field, p=p, locations=locations, node_weights=None)
    np.testing.assert_array_equal(weighted, unweighted)
