import numpy as np
import pytest
from _test_inputs.distance import *
from _test_inputs.readers import *

import earthkit.hydro as ekh


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
    dist = ekh.distance.array.max(
        river_network,
        stations_list,
        upstream=upstream,
        downstream=downstream,
        field=weights,
    )
    np.testing.assert_allclose(dist, result)
