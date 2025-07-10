from earthkit.hydro.utils.missing import is_missing


def _find_catchments_2D(field, did, uid, eid, mv, overwrite):
    """Updates field in-place with the value of its downstream nodes, dealing
    with missing values for 2D fields.

    Parameters
    ----------
    river_network : earthkit.hydro.network.RiverNetwork
        An earthkit-hydro river network object.
    field : numpy.ndarray
        The input field.
    grouping : numpy.ndarray
        The array of node indices.
    mv : scalar
        The missing value indicator.
    overwrite : bool
        If True, overwrite existing non-missing values in the field array.

    Returns
    -------
    None

    """
    down_not_missing = ~is_missing(field[..., uid], mv)
    did = did[
        down_not_missing
    ]  # only update nodes where the downstream belongs to a catchment
    if not overwrite:
        up_is_missing = is_missing(field[..., did], mv)
        did = did[up_is_missing]
    else:
        up_is_missing = None
    uid = (
        uid[down_not_missing][up_is_missing]
        if up_is_missing is not None
        else uid[down_not_missing]
    )
    field[..., did] = field[..., uid]
