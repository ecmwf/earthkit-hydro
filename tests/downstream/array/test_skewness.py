import numpy as np
import pytest
from _test_inputs.accumulation import *
from _test_inputs.readers import *

import earthkit.hydro as ekh


@pytest.mark.parametrize(
    "river_network, input_field, expected",
    [
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1c,
            downstream_metric_skewness_1c,
        ),
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1e,
            downstream_metric_skewness_1e,
        ),
    ],
    indirect=["river_network"],
)
@pytest.mark.parametrize("array_backend", ["numpy", "torch"])
def test_downstream_metric_skewness(river_network, input_field, expected, array_backend):
    """Test downstream skewness computation."""
    river_network = river_network.to_device("cpu", array_backend)
    xp = ekh._backends.find.get_array_backend(array_backend)
    output_field = ekh.downstream.array.skewness(
        river_network, xp.asarray(input_field), node_weights=None, return_type="masked"
    )
    output_field = np.asarray(output_field)
    expected_arr = np.asarray(expected)
    assert output_field.dtype == expected_arr.dtype
    np.testing.assert_allclose(output_field, expected_arr, rtol=1e-5, atol=1e-9, equal_nan=True)
