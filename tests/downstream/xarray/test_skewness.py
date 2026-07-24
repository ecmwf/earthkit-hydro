# SPDX-FileCopyrightText: 2026- European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import xarray as xr
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
def test_downstream_skewness_xarray(river_network, input_field, expected):
    """Test downstream skewness with xarray input."""
    field_da = xr.DataArray(
        input_field,
        dims=["node_index"],
        coords={"node_index": np.arange(len(input_field))},
    )
    result = ekh.downstream.skewness(river_network, field_da, return_type="masked")
    assert isinstance(result, xr.DataArray)
    expected_arr = np.asarray(expected)
    np.testing.assert_allclose(result.values, expected_arr, rtol=1e-5, atol=1e-9, equal_nan=True)
