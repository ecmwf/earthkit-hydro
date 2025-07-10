import numpy as np


def points_to_numpy(points):
    """
    Converts a list of tuples (indices) into a tuple of lists
    for use in numpy indexing.

    Parameters
    ----------
    points : list
        List of tuple indices of the points.

    Returns
    -------
    tuple
        Tuple of points suitable for numpy indexing.
    """
    # transform here list of tuples (indices) into a tuple of lists
    # (easier to manipulate)
    points = np.array(points)
    return (points[:, 0], points[:, 1])


def points_to_1d_indices(river_network, stations):
    """ "
    Converts a numpy index into a 1D index suitable
    for use with the flattened river representation.

    Parameters
    ----------
    river_network : earthkit.hydro.network.RiverNetwork
        The RiverNetwork instance calling the method.
    stations : tuple
        Tuple of numpy arrays defining the points.

    Returns
    -------
    numpy.ndarray
        1D array of indices.
    """
    node_numbers = np.cumsum(river_network.mask) - 1
    valid_stations = river_network.mask[stations]
    if np.any(~valid_stations):
        raise ValueError("Not all points are present on the river network.")
    stations = tuple(station_index[valid_stations] for station_index in stations)
    stations_1d = node_numbers.reshape(river_network.shape)[stations]
    return stations_1d
