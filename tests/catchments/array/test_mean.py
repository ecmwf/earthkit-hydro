import numpy as np
import pytest
from _test_inputs.catchment import *
from _test_inputs.accumulation import input_field_1c
from _test_inputs.readers import *

import earthkit.hydro as ekh


@pytest.mark.parametrize(
    "river_network, field, locations, expected",
    [
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1c,
            catchment_query_field_1,
            catchment_mean_1c,
        ),
    ],
    indirect=["river_network"],
)
@pytest.mark.parametrize("array_backend", ["numpy", "torch", "jax"])
def test_catchments_mean(river_network, field, locations, expected, array_backend):
    """Test catchment mean aggregation."""
    river_network = river_network.to_device("cpu", array_backend)
    xp = ekh._backends.find.get_array_backend(array_backend)
    result = ekh.catchments.array.mean(river_network, xp.asarray(field), locations=locations)
    result = np.asarray(result)
    print("Result:", result)
    print("Expected:", expected)
    np.testing.assert_allclose(result, expected, rtol=1e-6)
