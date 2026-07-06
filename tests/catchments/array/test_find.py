import numpy as np
import pytest
from _test_inputs.catchment import *
from _test_inputs.readers import *

import earthkit.hydro as ekh


@pytest.mark.parametrize(
    "river_network, query_field, find_catchments",
    [
        (
            ("cama_nextxy", cama_nextxy_1),
            catchment_query_field_1,
            catchment_1,
        ),
        (("cama_nextxy", cama_nextxy_2), catchment_query_field_2, catchment_2),
    ],
    indirect=["river_network"],
)
def test_find_catchments_2d(river_network, query_field, find_catchments):
    # field = np.zeros(river_network.mask.shape, dtype="int")
    # field[river_network.mask] = query_field
    network_find_catchments = ekh.catchments.array.find(river_network, locations=query_field)
    print(find_catchments)
    print(network_find_catchments)
    np.testing.assert_array_equal(network_find_catchments.flat[river_network.mask], find_catchments)
    # np.testing.assert_array_equal(network_find_catchments[~river_network.mask], 0)
