from earthkit.hydro._core.flow import propagate


def _flow_find(
    xp,
    river_network,
    field,
    overwrite=True,
    invert_graph=True,
):
    op = _find_catchments

    def operation(
        field,
        did,
        uid,
        eid,
    ):
        return op(
            xp,
            field,
            did,
            uid,
            eid,
            overwrite=overwrite,
        )

    field = propagate(
        river_network,
        river_network.groups,
        field,
        invert_graph,
        operation,
    )

    return field


def _find_catchments(xp, field, did, uid, eid, overwrite):
    """
    Updates field in-place with the value of its downstream nodes,
    dealing with missing values for 2D fields.

    Parameters
    ----------
    river_network : earthkit.hydro.network.RiverNetwork
        An earthkit-hydro river network object.
    field : numpy.ndarray
        The input field.
    grouping : numpy.ndarray
        The array of node indices.
    overwrite : bool
        If True, overwrite existing non-missing values in the field array.

    Returns
    -------
    None
    """
    down_not_missing = ~xp.isnan(xp.gather(field, uid, axis=-1))
    did = did[
        down_not_missing
    ]  # only update nodes where the downstream belongs to a catchment
    if not overwrite:
        up_is_missing = xp.isnan(xp.gather(field, did, axis=-1))
        did = did[up_is_missing]
    else:
        up_is_missing = None
    uid = (
        uid[down_not_missing][up_is_missing]
        if up_is_missing is not None
        else uid[down_not_missing]
    )
    updates = xp.gather(field, uid, axis=-1)
    return xp.scatter_assign(field, did, updates)
