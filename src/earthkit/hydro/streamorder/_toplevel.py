import earthkit.hydro.streamorder.array as array
from earthkit.hydro._utils.decorators import xarray


@xarray
def strahler(
    river_network,
    return_type=None,
    input_core_dims=None,
):
    return array.strahler(river_network=river_network, return_type=return_type)


@xarray
def shreve(
    river_network,
    return_type=None,
    input_core_dims=None,
):
    return array.shreve(river_network=river_network, return_type=return_type)
