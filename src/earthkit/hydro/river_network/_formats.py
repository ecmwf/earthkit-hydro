from io import BytesIO
from urllib.request import urlopen

import joblib
import numpy as np
import xarray as xr

from earthkit.hydro._readers import assign_coords, from_cama_nextxy, from_d8, from_grit
from earthkit.hydro._readers._cama import from_cama_nextxy_raw, load_cama_data
from earthkit.hydro._readers._d8 import from_d8_raw, load_d8_data


def _encode_da(da, mv):
    da.attrs["generated_by"] = "earthkit-hydro"
    da.encoding = {
        "_FillValue": mv,
    }
    return da


def _compute_offsets(river_network_storage):
    d, u, _ = river_network_storage.sorted_data
    mask = river_network_storage.mask
    coords = river_network_storage.coords

    if coords is None:
        raise ValueError("River network does not have coordinates.")

    ny, nx = river_network_storage.shape

    def shortest_offset(delta, n):
        delta = delta % n
        delta[delta > n // 2] -= n
        return delta

    dx = np.zeros(mask.shape, dtype=int)
    dx[u] = shortest_offset((mask[d] % nx) - (mask[u] % nx), nx)

    dy = np.zeros(mask.shape, dtype=int)
    dy[u] = shortest_offset((mask[d] // nx) - (mask[u] // nx), ny)

    return dx, dy, mask, coords


class Precomputed:
    def create(self, path, source):
        if source == "file":
            return joblib.load(path)
        elif source == "url":
            with urlopen(path) as response:
                return joblib.load(BytesIO(response.read()))
        else:
            raise ValueError(f"Unsupported source for precomputed river network format: {source}.")

    def export_to(self, river_network_storage, path, compression):
        joblib.dump(river_network_storage, path, compress=compression)


class CaMa:
    missing_value = -9999

    def create(self, path, source):
        data, coords = load_cama_data(path, "cama", source)
        river_network_storage = from_cama_nextxy(*data)
        return assign_coords(river_network_storage, data, coords)

    def load_partial(self, path, source):
        data, coords = load_cama_data(path, "cama", source)
        return from_cama_nextxy_raw(*data), coords

    def export_to(self, river_network_storage, path, compression):
        dx, dy, mask, coords = _compute_offsets(river_network_storage)
        shape = river_network_storage.shape
        mv = self.missing_value

        sinks = (dx == 0) & (dy == 0)
        cols, rows = np.indices(shape)
        rows = rows.flat[mask]
        cols = cols.flat[mask]

        x_masked = ((rows + dx) % shape[1]) + 1
        x_masked[sinks] = -9
        del dx, rows
        x = np.full(shape, mv, dtype=np.int32)
        x.flat[mask] = x_masked
        del x_masked

        y_masked = ((cols + dy) % shape[0]) + 1
        y_masked[sinks] = -9
        del dy, cols
        y = np.full(shape, mv, dtype=np.int32)
        y.flat[mask] = y_masked
        del y_masked

        coord1, coord2 = coords.keys()
        da_x = xr.DataArray(x, dims=(coord1, coord2), coords=coords, name="nextx")
        da_x = _encode_da(da_x, mv)
        da_y = xr.DataArray(y, dims=(coord1, coord2), coords=coords, name="nexty")
        da_y = _encode_da(da_y, mv)
        ds = xr.Dataset({"nextx": da_x, "nexty": da_y})
        ds.to_netcdf(path)


class D8:
    name = None
    missing_value = None
    lut = None

    def create(self, path, source):
        data, coords = load_d8_data(path, self.name, source)
        river_network_storage = from_d8(data, river_network_format=self.name)
        return assign_coords(river_network_storage, data, coords)

    def load_partial(self, path, source):
        data, coords = load_d8_data(path, self.name, source)
        return from_d8_raw(data, river_network_format=self.name), coords

    def export_to(self, river_network_storage, path, compression):
        dx, dy, mask, coords = _compute_offsets(river_network_storage)
        shape = river_network_storage.shape

        if not (np.all(np.abs(dx) <= 1) and np.all(np.abs(dy) <= 1)):
            raise ValueError("River network is not representable in d8 format.")

        data = np.full(shape, self.missing_value, dtype=np.uint8)
        try:
            data.flat[mask] = self.lut[dy + 1, dx + 1]
        except Exception as e:
            raise ValueError("Failed to represent river network as D8") from e

        coord1, coord2 = coords.keys()
        da = xr.DataArray(data.astype(np.uint8), dims=(coord1, coord2), coords=coords, name="ldd")
        da = _encode_da(da, self.missing_value)
        da.to_netcdf(path)


class PCRD8(D8):
    name = "pcr_d8"
    missing_value = 255
    lut = np.array(
        [
            [7, 8, 9],  # dy = -1
            [4, 5, 6],  # dy =  0
            [1, 2, 3],  # dy = +1
        ],
        dtype=np.uint8,
    )


class ESRID8(D8):
    name = "esri_d8"
    missing_value = 255
    lut = np.array(
        [
            [32, 64, 128],  # dy = -1
            [16, 0, 1],  # dy =  0
            [8, 4, 2],  # dy = +1
        ],
        dtype=np.uint8,
    )


class MeritD8(ESRID8):
    name = "merit_d8"
    missing_value = 247


class Grit:
    def create(self, path, source):
        assert path.endswith(".gpkg")
        return from_grit(path)


FORMATS = {
    "precomputed": Precomputed(),
    "cama": CaMa(),
    "pcr_d8": PCRD8(),
    "esri_d8": ESRID8(),
    "merit_d8": MeritD8(),
    "grit": Grit(),
}
