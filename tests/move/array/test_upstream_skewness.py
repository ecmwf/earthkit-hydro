import numpy as np
import pytest
from _test_inputs.movement import *
from _test_inputs.readers import *

import earthkit.hydro as ekh


@pytest.mark.parametrize(
    "river_network, expected",
    [
        (("cama_nextxy", cama_nextxy_1), move_upstream_skewness_1),
        (("cama_nextxy", cama_nextxy_2), move_upstream_skewness_2),
    ],
    indirect=["river_network"],
)
@pytest.mark.parametrize("array_backend", ["numpy", "torch"])
def test_move_upstream_skewness(river_network, expected, array_backend):
    """Test move upstream skewness computation."""
    river_network = river_network.to_device("cpu", array_backend)
    xp = ekh._backends.find.get_array_backend(array_backend)
    field = xp.arange(1, river_network.n_nodes + 1, dtype=xp.float64)
    output_field = ekh.move.array.upstream(
        river_network, field, metric="skewness", return_type="masked"
    )
    output_field = np.asarray(output_field)
    expected_arr = np.asarray(expected)
    np.testing.assert_allclose(output_field, expected_arr, rtol=1e-5, atol=1e-9, equal_nan=True)
