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
            catchment_sum_1c,
        ),
    ],
    indirect=["river_network"],
)
def test_catchments_sum(river_network, field, locations, expected):
    """Test catchment sum aggregation."""
    result = ekh.catchments.array.sum(river_network, field, locations=locations)
    print("Result:", result)
    print("Expected:", expected)
    np.testing.assert_allclose(result, expected, rtol=1e-6)
