from earthkit.hydro.upstream import array
from earthkit.hydro._utils.decorators import xarray

@xarray
def upstream_area(river_network, return_type=None):
    if river_network._storage.area is None:
        raise ValueError("River network pixel area is not available.")

    return array.sum(river_network, field=river_network._storage.area, return_type=return_type)