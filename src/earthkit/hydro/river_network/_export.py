import numpy as np
import xarray as xr

from earthkit.hydro.data_structures._network_storage import RiverNetworkStorage


def export(
    river_network_storage: RiverNetworkStorage, river_network_format: str, fpath: str
):
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
    downx = np.full(shape, dtype=int, fill_value=-1)
    downy = np.full(shape, dtype=int, fill_value=-1)
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
        data = lut[downy + 1, downx + 1]
    elif river_network_format == "esri_d8" or river_network_format == "merit_d8":
        lut = np.array(
            [
                [32, 64, 128],  # dy = -1
                [16, 0, 1],  # dy =  0
                [8, 4, 2],  # dy = +1
            ],
            dtype=np.uint8,
        )
        data = lut[downy + 1, downx + 1]
    else:
        raise ValueError(
            f"Unsupported river network format for the export method: {river_network_format}."
        )

    coord1, coord2 = coords.keys()
    da = xr.DataArray(data, dims=(coord1, coord2), coords=coords, name="ldd")
    da.attrs["generated_by"] = "earthkit-hydro"
    da.to_netcdf(fpath)
