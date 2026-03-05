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
            distance_1_to_source_shortest,
        ),
    ],
    indirect=["river_network"],
)
@pytest.mark.parametrize("array_backend", ["numpy", "torch", "jax"])
def test_distance_to_source(river_network, field, expected, array_backend):
    """Test distance to source computation."""
    river_network = river_network.to_device("cpu", array_backend)
    result = ekh.distance.array.to_source(
        river_network, field=field, path="shortest", return_type="masked"
    )
    result = np.asarray(result)
    print("Result:", result)
    print("Expected:", expected)
    np.testing.assert_allclose(result, expected, rtol=1e-6)
