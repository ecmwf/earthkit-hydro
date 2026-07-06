import numpy as np
import pytest
from _test_inputs.accumulation import input_field_1b
from _test_inputs.catchment import *
from _test_inputs.readers import *

import earthkit.hydro as ekh

try:
    from earthkit.hydro import _rust  # noQA: F401

    RUST = True
except Exception:
    RUST = False


@pytest.mark.skipif(not RUST, reason="Rust unavailable")
@pytest.mark.parametrize(
    "river_network, field, locations, expected",
    [
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1b,
            catchment_query_field_1,
            catchment_mode_1b,
        ),
    ],
    indirect=["river_network"],
)
@pytest.mark.parametrize("array_backend", ["numpy"])  # Mode only supported for numpy
def test_catchments_mode(river_network, field, locations, expected, array_backend):
    """Test catchment mode aggregation."""
    river_network = river_network.to_device("cpu", array_backend)
    xp = ekh._backends.find.get_array_backend(array_backend)
    result = ekh.catchments.array.mode(river_network, xp.asarray(field), locations=locations)
    result = np.asarray(result)
    print("Result:", result)
    print("Expected:", expected)
    np.testing.assert_array_equal(result, expected)
