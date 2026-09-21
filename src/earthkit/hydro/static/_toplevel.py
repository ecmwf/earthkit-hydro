from earthkit.hydro import upstream
from earthkit.hydro._utils.decorators import xarray

from .array import area as array_area


@xarray
def area(river_network, return_type=None):
    return array_area(river_network, return_type=return_type)


@xarray
def upstream_area(river_network, return_type=None):
    if river_network._storage.area is None:
        raise ValueError("River network pixel area is not available.")

    return upstream.array.sum(river_network, field=river_network._storage.area, return_type=return_type)
