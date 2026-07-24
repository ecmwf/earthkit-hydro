# SPDX-FileCopyrightText: 2026- European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from _test_inputs.accumulation import input_field_1c
from _test_inputs.catchment import *
from _test_inputs.readers import *

import earthkit.hydro as ekh


@pytest.mark.parametrize(
    "river_network, field, locations, expected",
    [
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1c,
            catchment_query_field_1,
            catchment_skewness_1c,
        ),
    ],
    indirect=["river_network"],
)
@pytest.mark.parametrize("array_backend", ["numpy", "torch"])
def test_catchments_skewness(river_network, field, locations, expected, array_backend):
    """Test catchment skewness aggregation."""
    river_network = river_network.to_device("cpu", array_backend)
    xp = ekh._backends.find.get_array_backend(array_backend)
    result = ekh.catchments.array.skewness(river_network, xp.asarray(field), locations=locations)
    result = np.asarray(result)
    expected_arr = np.asarray(expected)
    assert result.shape[-1] == np.asarray(locations).shape[-1]
    np.testing.assert_allclose(result, expected_arr, rtol=1e-5, atol=1e-9, equal_nan=True)
