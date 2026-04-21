import numpy as np
import pytest
from _test_inputs.accumulation import *
from _test_inputs.readers import *

import earthkit.hydro as ekh


@pytest.mark.parametrize(
    "river_network, input_field, mv",
    [
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1c,
            mv_1c,
        ),
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1e,
            mv_1e,
        ),
    ],
    indirect=["river_network"],
)
def test_downstream_metric_std(river_network, input_field, mv):
    """Test downstream standard deviation computation."""
    output_field = ekh.downstream.array.std(
        river_network, input_field, node_weights=None, return_type="masked"
    )

    # Std should be non-negative
    assert np.all(output_field[~np.isnan(output_field)] >= 0)

    # Test that std is 0 for uniform fields
    uniform_field = np.ones(river_network.n_nodes)
    std_uniform = ekh.downstream.array.std(
        river_network, uniform_field, node_weights=None, return_type="masked"
    )
    np.testing.assert_allclose(std_uniform, 0, atol=1e-10)

    # Test relationship with variance
    var_field = ekh.downstream.array.var(
        river_network, input_field, node_weights=None, return_type="masked"
    )
    np.testing.assert_allclose(output_field**2, var_field, rtol=1e-10, equal_nan=True)
