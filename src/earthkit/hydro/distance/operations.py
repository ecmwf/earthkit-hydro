import numpy as np
import xarray as xr

import earthkit.hydro.distance.array._operations as array
from earthkit.hydro.backends.find import get_array_backend
from earthkit.hydro.catchments.array.operations import preprocess_stations
from earthkit.hydro.utils.decorators import xarray


def convert_locations(func):
    def wrapper(river_network, field, locations, *args, **kwargs):
        xp = get_array_backend(river_network.groups[0])
        orig_locations = locations
        xr_present = isinstance(field, (xr.DataArray, xr.Dataset))
        dict_locations = isinstance(locations, dict)
        if dict_locations:
            assert xr_present

            lats = field.lat.data
            lons = field.lon.data

            # TODO: decide if acceptable to always use np here
            locations = []
            for lat_val, lon_val in orig_locations.values():
                ilat = np.abs(lats - lat_val).argmin()
                ilon = np.abs(lons - lon_val).argmin()
                locations.append((int(ilat), int(ilon)))

        locations = xp.asarray(locations, device=river_network.device)
        stations_1d = preprocess_stations(xp, river_network, locations)

        result = func(river_network, field, stations_1d, *args, **kwargs)

        return result

    return wrapper


@convert_locations
@xarray
def min(river_network, field, locations, upstream=False, downstream=True):
    return array.min(river_network, field, locations, upstream, downstream)


@convert_locations
@xarray
def max(river_network, field, locations, upstream=False, downstream=True):
    return array.max(river_network, field, locations, upstream, downstream)


def to_source(*args, **kwargs):
    raise NotImplementedError


def to_sink(*args, **kwargs):
    raise NotImplementedError
