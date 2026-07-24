# SPDX-FileCopyrightText: 2026- European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import xarray as xr
from _test_inputs.readers import *

import earthkit.hydro as ekh


def generate_ldd(data, mv):
    if isinstance(data, tuple):
        x, y = data

        coords = {"lat": np.arange(x.shape[0]), "lon": np.arange(x.shape[1])}

        coord1, coord2 = coords.keys()
        da_x = xr.DataArray(x, dims=(coord1, coord2), coords=coords, name="nextx")
        da_x.encoding = {
            "_FillValue": mv,
        }
        da_y = xr.DataArray(y, dims=(coord1, coord2), coords=coords, name="nexty")
        da_y.encoding = {
            "_FillValue": mv,
        }
        return xr.Dataset({"nextx": da_x, "nexty": da_y})

    else:
        coords = {"lat": np.arange(data.shape[0]), "lon": np.arange(data.shape[1])}
        coord1, coord2 = coords
        da = xr.DataArray(data.astype(np.int32), dims=(coord1, coord2), coords=coords, name="ldd")
        da.encoding = {
            "_FillValue": mv,
        }
        return da


@pytest.mark.parametrize(
    "ldd, mv, river_network_format",
    [
        (d8_ldd_1, 255, "pcr_d8"),
        (cama_nextxy_1, -9, "cama"),
        (d8_ldd_2, 255, "pcr_d8"),
        (cama_nextxy_2, -9, "cama"),
    ],
)
def test_create_export(tmp_path, ldd, mv, river_network_format):
    original = str(tmp_path / "original.nc")
    ds = generate_ldd(ldd, mv)
    ds.to_netcdf(original)

    net = ekh.river_network.create(original, river_network_format)

    original_sum = ekh.upstream.sum(net, np.arange(net.n_nodes))

    exported_pcr = str(tmp_path / "exported_pcr.nc")
    exported_cama = str(tmp_path / "exported_cama.nc")
    exported_merit = str(tmp_path / "exported_merit.nc")
    exported_esri = str(tmp_path / "exported_esri.nc")

    export_options = [
        (exported_pcr, "pcr_d8"),
        (exported_cama, "cama"),
        (exported_merit, "merit_d8"),
        (exported_esri, "esri_d8"),
    ]

    for file, fmt in export_options:
        ekh.river_network.export(net, file, fmt)
        net = ekh.river_network.create(file, fmt)
        exported_sum = ekh.upstream.sum(net, np.arange(net.n_nodes))
        np.testing.assert_array_equal(original_sum, exported_sum)
