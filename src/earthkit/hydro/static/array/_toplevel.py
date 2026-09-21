from earthkit.hydro._backends.numpy_backend import NumPyBackend
from earthkit.hydro._utils.decorators import mask
from earthkit.hydro.static.array import _operations
from earthkit.hydro.upstream import array


def area(river_network, return_type=None):
    if river_network._storage.area is None:
        raise ValueError("River network pixel area is not available.")
    return_type = river_network.return_type if return_type is None else return_type
    if return_type not in ["gridded", "masked"]:
        raise ValueError("return_type must be either 'gridded' or 'masked'.")
    func = mask(return_type == "gridded")(_operations.area)
    return func(river_network=river_network, xp=NumPyBackend(), field=river_network._storage.area)


def upstream_area(river_network, return_type=None):
    if river_network._storage.area is None:
        raise ValueError("River network pixel area is not available.")
    return array.sum(river_network, field=river_network._storage.area, return_type=return_type)
