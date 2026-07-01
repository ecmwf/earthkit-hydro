from typing import Union

import numpy as np
import xarray as xr

from earthkit.hydro.data_structures import RiverNetwork
from earthkit.hydro.data_structures._network_storage import RiverNetworkStorage


def export(
    river_network: Union[RiverNetworkStorage, RiverNetwork],
    path: str,
    river_network_format: str = "precomputed",
    compression=1,
):
    if river_network_format not in ["precomputed", "pcr_d8", "esri_d8", "merit_d8"]:
        raise ValueError(
            f"Exporting river network to format {river_network_format} is not currently supported."
        )

    river_network_storage = (
        river_network
        if isinstance(river_network, RiverNetworkStorage)
        else river_network._storage
    )

    if river_network_format == "precomputed":
        import joblib

        joblib.dump(river_network_storage, path, compress=compression)
        return

    missing_values = {"pcr_d8": 255, "esri_d8": 255, "merit_d8": 247}

    d, u, _ = river_network_storage.sorted_data
    mask = river_network_storage.mask
    coords = river_network_storage.coords

    shape = river_network_storage.shape
    _, nx = shape

    dx = np.zeros(mask.shape, dtype=int)
    dy = np.zeros(mask.shape, dtype=int)
    dx[u] = (mask[d] % nx) - (mask[u] % nx)
    dy[u] = (mask[d] // nx) - (mask[u] // nx)

    # Scatter back to 2D grids
    mv = missing_values[river_network_format]
    downx = np.full(shape, dtype=int, fill_value=mv)
    downy = np.full(shape, dtype=int, fill_value=mv)
    downx.flat[mask] = dx
    downy.flat[mask] = dy

    if river_network_format == "pcr_d8":
        lut = np.array(
            [
                [7, 8, 9],  # dy = -1
                [4, 5, 6],  # dy =  0
                [1, 2, 3],  # dy = +1
            ],
            dtype=np.uint8,
        )
        try:
            data = lut[downy + 1, downx + 1]
        except Exception as e:
            raise ValueError("Failed to represent river network as D8") from e
    elif river_network_format == "esri_d8" or river_network_format == "merit_d8":
        lut = np.array(
            [
                [32, 64, 128],  # dy = -1
                [16, 0, 1],  # dy =  0
                [8, 4, 2],  # dy = +1
            ],
            dtype=np.uint8,
        )
        try:
            data = lut[downy + 1, downx + 1]
        except Exception as e:
            raise ValueError("Failed to represent river network as D8") from e
    else:
        raise ValueError(
            f"Unsupported river network format for the export method: {river_network_format}."
        )

    coord1, coord2 = coords.keys()
    da = xr.DataArray(
        data.astype(np.uint8), dims=(coord1, coord2), coords=coords, name="ldd"
    )
    da.attrs["generated_by"] = "earthkit-hydro"
    da.encoding = {
        "_FillValue": mv,  # Map np.nan to -9999 on disk
    }
    da.to_netcdf(path)
