import numpy as np
import pytest
import xarray as xr
from _test_inputs.accumulation import input_field_1c, upstream_metric_mean_1c
from _test_inputs.readers import cama_nextxy_1

import earthkit.hydro as ekh

pytestmark = pytest.mark.parametrize(
    "river_network",
    [("cama_nextxy", cama_nextxy_1)],
    indirect=True,
)

# All three encode the same 5 outlets in the same insertion order
_OUTLET_NODES = [11, 8, 13, 10, 12]
_OUTLET_NODES_2D = [(2, 1), (1, 3), (2, 3), (2, 0), (2, 2)]
_OUTLET_DICT = {"B": (2, 1), "E": (1, 3), "D": (2, 3), "A": (2, 0), "C": (2, 2)}
_OUTLET_MEANS = upstream_metric_mean_1c[np.array(_OUTLET_NODES)]
_FIELD_DA = xr.DataArray(
    input_field_1c.copy(),
    dims=["node_index"],
    coords={"node_index": np.arange(len(input_field_1c))},
)


def test_catchment_mean_preserves_location_order_list(river_network):
    result = ekh.catchments.mean(river_network, _FIELD_DA, locations=_OUTLET_NODES)

    np.testing.assert_array_equal(result.coords["node_index"].values, _OUTLET_NODES)
    np.testing.assert_allclose(result.values, _OUTLET_MEANS, rtol=1e-6)


def test_catchment_mean_preserves_location_order_2d(river_network):
    result = ekh.catchments.mean(river_network, _FIELD_DA, locations=_OUTLET_NODES_2D)

    np.testing.assert_array_equal(result.coords["node_index"].values, _OUTLET_NODES)
    np.testing.assert_array_equal(result.coords["y"].values, [t[0] for t in _OUTLET_NODES_2D])
    np.testing.assert_array_equal(result.coords["x"].values, [t[1] for t in _OUTLET_NODES_2D])
    np.testing.assert_allclose(result.values, _OUTLET_MEANS, rtol=1e-6)


def test_catchment_mean_preserves_location_order_dict(river_network):
    result = ekh.catchments.mean(river_network, _FIELD_DA, locations=_OUTLET_DICT)

    np.testing.assert_array_equal(result.coords["node_index"].values, _OUTLET_NODES)
    np.testing.assert_array_equal(result.coords["name"].values, list(_OUTLET_DICT.keys()))
    np.testing.assert_allclose(result.values, _OUTLET_MEANS, rtol=1e-6)
