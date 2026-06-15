import numpy as np
import pytest
import xarray as xr
from _test_inputs.accumulation import *
from _test_inputs.catchment import *
from _test_inputs.readers import *

import earthkit.hydro as ekh

try:
    from earthkit.hydro import _rust  # noQA: F401

    RUST = True
except Exception:
    RUST = False


def make_river_network_with_coords(flow_directions):
    from earthkit.hydro._readers import from_cama_nextxy
    from earthkit.hydro.data_structures import RiverNetwork

    rn = RiverNetwork(from_cama_nextxy(*flow_directions))
    # Assign synthetic grid coords so the xarray wrapper can attach them to output
    ny, nx = rn.shape
    rn.coords = {"y": np.arange(ny), "x": np.arange(nx)}
    return rn


def field_to_xarray(river_network, field_1d):
    """Convert a 1-D masked field to a 2-D DataArray with y/x coords."""
    ny, nx = river_network.shape
    field_2d = np.zeros((ny, nx), dtype=field_1d.dtype)
    field_2d.flat[river_network.mask] = field_1d
    return xr.DataArray(
        field_2d,
        dims=["y", "x"],
        coords={"y": river_network.coords["y"], "x": river_network.coords["x"]},
    )


@pytest.mark.skipif(not RUST, reason="Rust unavailable")
@pytest.mark.parametrize(
    "flow_directions, input_field, locations, expected, p",
    [
        (
            cama_nextxy_1,
            input_field_1c,
            catchment_query_field_1,
            catchment_percentile_p05_1c,
            0.5,
        ),
        (
            cama_nextxy_1,
            input_field_1c,
            catchment_query_field_1,
            catchment_percentile_p025_1c,
            0.25,
        ),
    ],
)
def test_catchments_percentile_xarray_unweighted(flow_directions, input_field, locations, expected, p):
    rn = make_river_network_with_coords(flow_directions)
    field_xr = field_to_xarray(rn, input_field)

    result = ekh.catchments.percentile(rn, field_xr, p=p, locations=locations)
    assert isinstance(result, xr.DataArray)
    np.testing.assert_allclose(result.values, expected)


@pytest.mark.skipif(not RUST, reason="Rust unavailable")
@pytest.mark.parametrize(
    "flow_directions, input_field, locations, expected",
    [
        (
            cama_nextxy_1,
            input_field_1c,
            catchment_query_field_1,
            catchment_percentile_weighted_p05_1c,
        ),
    ],
)
def test_catchments_percentile_xarray_weighted(flow_directions, input_field, locations, expected):
    rn = make_river_network_with_coords(flow_directions)
    field_xr = field_to_xarray(rn, input_field)
    ny, nx = rn.shape
    weights_2d = np.zeros((ny, nx), dtype="float64")
    weights_2d.flat[rn.mask] = np.arange(1, rn.n_nodes + 1, dtype="float64")
    weights_xr = xr.DataArray(
        weights_2d,
        dims=["y", "x"],
        coords={"y": rn.coords["y"], "x": rn.coords["x"]},
    )

    result = ekh.catchments.percentile(rn, field_xr, p=0.5, locations=locations, node_weights=weights_xr)
    assert isinstance(result, xr.DataArray)
    np.testing.assert_allclose(result.values, expected)
