import earthkit.hydro.streamorder.array._operations as array


def shreve(river_network, return_type=None):
    return_type = river_network.return_type if return_type is None else return_type
    return array.shreve(river_network=river_network, return_type=return_type)


def strahler(river_network, return_type=None):
    return_type = river_network.return_type if return_type is None else return_type
    return array.strahler(river_network=river_network, return_type=return_type)
