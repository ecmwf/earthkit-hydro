import numpy as np


def extend_array(arr, extra_shape):
    current_shape = arr.shape
    new_shape = (*extra_shape, *current_shape)
    extended_array = np.broadcast_to(arr, new_shape)
    return extended_array


def convert_to_2d(river_network, array, fill_value):
    field = np.full(river_network.mask.shape, fill_value=fill_value, dtype=array.dtype)
    field[river_network.mask] = array
    return field
