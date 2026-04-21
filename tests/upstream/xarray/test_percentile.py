import numpy as np
import pytest
import xarray as xr
from _test_inputs.accumulation import *
from _test_inputs.readers import *

import earthkit.hydro as ekh


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


@pytest.mark.parametrize(
    "flow_directions, input_field, expected_masked, expected_gridded, p",
    [
        (
            cama_nextxy_1,
            input_field_1c,
            upstream_metric_percentile_p05_1c,
            0.5,
        ),
        (
            cama_nextxy_1,
            input_field_1c,
            upstream_metric_percentile_p025_1c,
            0.25,
        ),
    ],
)
def test_upstream_percentile_xarray_masked(
    flow_directions, input_field, expected_masked, p
):
    rn = make_river_network_with_coords(flow_directions)
    field_xr = field_to_xarray(rn, input_field)

    # masked return type
    result = ekh.upstream.percentile(rn, field_xr, p=p, return_type="masked")
    assert isinstance(result, xr.DataArray)
    np.testing.assert_allclose(result.values, expected_masked)


@pytest.mark.parametrize(
    "flow_directions, input_field, expected_gridded, p",
    [
        (
            cama_nextxy_1,
            input_field_1c,
            upstream_metric_percentile_gridded_p05_1c,
            0.5,
        ),
    ],
)
def test_upstream_percentile_xarray_gridded(
    flow_directions, input_field, expected_gridded, p
):
    rn = make_river_network_with_coords(flow_directions)
    field_xr = field_to_xarray(rn, input_field)

    # gridded return type
    result = ekh.upstream.percentile(rn, field_xr, p=p, return_type="gridded")
    assert isinstance(result, xr.DataArray)
    np.testing.assert_allclose(result.values, expected_gridded)
