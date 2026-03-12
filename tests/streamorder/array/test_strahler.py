import numpy as np
import pytest
from _test_inputs.readers import *
from _test_inputs.streamorder import *

import earthkit.hydro as ekh


@pytest.mark.parametrize(
    "river_network, result",
    [
        (
            ("cama_nextxy", cama_nextxy_1),
            strahler_1,
        ),
        (
            ("cama_nextxy", cama_nextxy_2),
            strahler_2,
        ),
    ],
    indirect=["river_network"],
)
def test_streamorder_strahler(river_network, result):
    streamorder = ekh.streamorder.array.strahler(river_network, return_type="masked")
    np.testing.assert_allclose(streamorder, result)
