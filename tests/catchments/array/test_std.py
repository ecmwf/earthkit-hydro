import numpy as np
import pytest
from _test_inputs.catchment import *
from _test_inputs.accumulation import input_field_1c
from _test_inputs.readers import *

import earthkit.hydro as ekh


@pytest.mark.parametrize(
    "river_network, field, locations",
    [
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1c,
            catchment_query_field_1,
        ),
    ],
    indirect=["river_network"],
)
def test_catchments_std(river_network, field, locations):
    """Test catchment standard deviation aggregation."""
    result = ekh.catchments.array.std(river_network, field, locations=locations)

    # Std should be non-negative
    assert np.all(result >= 0)

    # Test that std is 0 for uniform fields
    uniform_field = np.ones(river_network.n_nodes)
    std_uniform = ekh.catchments.array.std(river_network, uniform_field, locations=locations)
    np.testing.assert_allclose(std_uniform, 0, atol=1e-10)

    # Test relationship with variance
    var_result = ekh.catchments.array.var(river_network, field, locations=locations)
    np.testing.assert_allclose(result**2, var_result, rtol=1e-10)
