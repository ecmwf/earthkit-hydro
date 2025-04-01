import numpy as np
import pytest
from test_inputs.distance import *
from test_inputs.readers import *

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
            distance_1_min_up_down,
        ),
        (
            ("cama_nextxy", cama_nextxy_1),
            stations,
            True,
            False,
            weights_1,
            distance_1_min_up,
        ),
        (
            ("cama_nextxy", cama_nextxy_1),
            stations,
            False,
            True,
            weights_1,
            distance_1_min_down,
        ),
    ],
    indirect=["river_network"],
)
def test_distance_min(
    river_network, stations_list, upstream, downstream, weights, result
):
    dist = ekh.distance.min(
        river_network,
        stations_list,
        upstream=upstream,
        downstream=downstream,
        weights=weights,
    )
    np.testing.assert_allclose(dist, result)


@pytest.mark.parametrize(
    "river_network, stations_list, upstream, downstream, weights, result",
    [
        (
            ("cama_nextxy", cama_nextxy_1),
            stations,
            True,
            False,
            weights_1,
            distance_1_max_up,
        ),
        (
            ("cama_nextxy", cama_nextxy_1),
            stations,
            False,
            True,
            weights_1,
            distance_1_max_down,
        ),
    ],
    indirect=["river_network"],
)
def test_distance_max(
    river_network, stations_list, upstream, downstream, weights, result
):
    dist = ekh.distance.max(
        river_network,
        stations_list,
        upstream=upstream,
        downstream=downstream,
        weights=weights,
    )
    np.testing.assert_allclose(dist, result)


@pytest.mark.parametrize(
    "river_network, stations_list, upstream, downstream, weights, result",
    [
        (
            ("cama_nextxy", cama_nextxy_1),
            stations,
            True,
            True,
            weights_1,
            length_1_min_up_down,
        ),
        (
            ("cama_nextxy", cama_nextxy_1),
            stations,
            True,
            False,
            weights_1,
            length_1_min_up,
        ),
        (
            ("cama_nextxy", cama_nextxy_1),
            stations,
            False,
            True,
            weights_1,
            length_1_min_down,
        ),
    ],
    indirect=["river_network"],
)
def test_length_min(
    river_network, stations_list, upstream, downstream, weights, result
):
    length = ekh.length.min(
        river_network,
        stations_list,
        upstream=upstream,
        downstream=downstream,
        weights=weights,
    )
    np.testing.assert_allclose(length, result)


@pytest.mark.parametrize(
    "river_network, stations_list, upstream, downstream, weights, result",
    [
        (
            ("cama_nextxy", cama_nextxy_1),
            stations,
            True,
            False,
            weights_1,
            length_1_max_up,
        ),
        (
            ("cama_nextxy", cama_nextxy_1),
            stations,
            False,
            True,
            weights_1,
            length_1_max_down,
        ),
    ],
    indirect=["river_network"],
)
def test_length_max(
    river_network, stations_list, upstream, downstream, weights, result
):
    length = ekh.length.max(
        river_network,
        stations_list,
        upstream=upstream,
        downstream=downstream,
        weights=weights,
    )
    np.testing.assert_allclose(length, result)
