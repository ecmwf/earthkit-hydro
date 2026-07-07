import numpy as np
import xarray as xr

from earthkit.hydro.data_structures import RiverNetwork
from earthkit.hydro.data_structures._network_storage import RiverNetworkStorage


def export(
    river_network: RiverNetworkStorage | RiverNetwork,
    path: str,
    river_network_format: str = "precomputed",
    compression=1,
):
    """
    Export a river network to a local file.

    .. note::
        Exporting to precomputed format is highly recommended for efficiency reasons.
        Other river network formats should only be used if compatibility is required with other tooling.

    .. note::
        For formats other than precomputed, only exporting as netcdf is currently supported.

    Parameters
    ----------
    river_network : RiverNetworkStorage | RiverNetwork
        The river network to export.
    path : str
        Where to export the river network.
    river_network_format : str
        The format of the river network data.
        Currently supported formats are "pcr_d8", "esri_d8"
        and "merit_d8".
    compression : int
        The compression factor to use for the saved file. Only applied if river_network_format is precomputed.

    Returns
    -------
    None. Writes the river network to a local file at `path`.
    """
    if river_network_format not in {"precomputed", "pcr_d8", "esri_d8", "merit_d8"}:
        raise ValueError(f"Exporting river network to format {river_network_format} is not currently supported.")

    river_network_storage = river_network if isinstance(river_network, RiverNetworkStorage) else river_network._storage

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

    if river_network_format in {"pcr_d8", "esri_d8", "merit_d8"} and not (
        np.all(np.abs(dx) <= 1) and np.all(np.abs(dy) <= 1)
    ):
        raise ValueError("River network is not representable in d8 format.")

    # Scatter back to 2D grids
    mv = missing_values[river_network_format]
    data = np.full(shape, mv, dtype=np.uint8)

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
            data.flat[mask] = lut[dy + 1, dx + 1]
        except Exception as e:
            raise ValueError("Failed to represent river network as D8") from e
    elif river_network_format in {"esri_d8", "merit_d8"}:
        lut = np.array(
            [
                [32, 64, 128],  # dy = -1
                [16, 0, 1],  # dy =  0
                [8, 4, 2],  # dy = +1
            ],
            dtype=np.uint8,
        )
        try:
            data.flat[mask] = lut[dy + 1, dx + 1]
        except Exception as e:
            raise ValueError("Failed to represent river network as D8") from e
    else:
        raise ValueError(f"Unsupported river network format for the export method: {river_network_format}.")

    coord1, coord2 = coords.keys()
    da = xr.DataArray(data.astype(np.uint8), dims=(coord1, coord2), coords=coords, name="ldd")
    da.attrs["generated_by"] = "earthkit-hydro"
    da.encoding = {
        "_FillValue": mv,
    }
    da.to_netcdf(path)
