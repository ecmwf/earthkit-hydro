from .utils import mask_and_unmask_data, is_missing


@mask_and_unmask_data
def catchment(river_network, field, mv=0, overwrite=True):
    """
    Propagates a field upstream to find catchments.

    Parameters
    ----------
    field : numpy.ndarray
        The input field to propagate.
    mv : int, optional
        The missing value to use (default is 0).
    overwrite : bool, optional
        If True, overwrites existing values (default is True).

    Returns
    -------
    numpy.ndarray
        The catchment field.
    """
    for group in river_network.topological_groups[:-1][::-1]:  # exclude sinks and invert topological ordering
        valid_group = group[
            ~is_missing(field[river_network.downstream_nodes[group]], mv)
        ]  # only update nodes where the downstream belongs to a catchment
        if not overwrite:
            valid_group = valid_group[is_missing(field[valid_group], mv)]
        field[valid_group] = field[river_network.downstream_nodes[valid_group]]
    return field


def subcatchment(river_network, field, mv=0):
    """
    Propagates a field upstream to find subcatchments.

    Parameters
    ----------
    field : numpy.ndarray
        The input field to propagate.
    mv : int, optional
        The missing value to use (default is 0).

    Returns
    -------
    numpy.ndarray
        The propagated subcatchment field.
    """
    return catchment(river_network, field, mv=mv, overwrite=False)
