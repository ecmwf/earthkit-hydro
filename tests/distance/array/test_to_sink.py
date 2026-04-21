import numpy as np
import pytest
from _test_inputs.distance import *
from _test_inputs.readers import *

import earthkit.hydro as ekh


@pytest.mark.parametrize(
    "river_network, field, expected",
    [
        (
            ("cama_nextxy", cama_nextxy_1),
            None,
            distance_1_to_sink_shortest,
        ),
    ],
    indirect=["river_network"],
)
def test_distance_to_sink(river_network, field, expected):
    """Test distance to sink computation."""
    result = ekh.distance.array.to_sink(
        river_network, field=field, path="shortest", return_type="masked"
    )
    print("Result:", result)
    print("Expected:", expected)
    np.testing.assert_array_equal(result, expected)
