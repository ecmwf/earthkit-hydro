def _ufunc_to_downstream(
    field,
    did,
    uid,
    eid,
    node_additive_weight,
    node_multiplicative_weight,
    node_modifier_use_upstream,
    edge_additive_weight,
    edge_multiplicative_weight,
    func,
    xp,
):
    """
    Updates field in-place by applying a ufunc at the downstream nodes
    of the grouping.

    Parameters
    ----------
    river_network : earthkit.hydro.network.RiverNetwork
        An earthkit-hydro river network object.
    field : numpy.ndarray
        The input field.
    grouping : numpy.ndarray
        An array of indices.
    mv : scalar
        A missing value indicator (not used in the function but kept for consistency).
    additive_weight : numpy.ndarray, optional
        A weight to be added to the field values. Default is None.
    multiplicative_weight : numpy.ndarray, optional
        A weight to be multiplied with the field values. Default is None.
    modifier_use_upstream : bool, optional
        If True, the modifiers are used on the upstream nodes instead of downstream.
        Default is True.
    ufunc : numpy.ufunc
        A universal function from the numpy library to be applied to the field data.
        Available ufuncs: https://numpy.org/doc/2.2/reference/ufuncs.html.
        Note: must allow two operands.

    Returns
    -------
    None
    """
    modifier_group = uid if node_modifier_use_upstream else did

    modifier_field = xp.gather(field, uid, axis=-1)
    if node_multiplicative_weight is not None:
        modifier_field *= node_multiplicative_weight[..., modifier_group]
    if edge_multiplicative_weight is not None:
        modifier_field *= edge_multiplicative_weight[..., eid]
    if node_additive_weight is not None:
        modifier_field += node_additive_weight[..., modifier_group]
    if edge_additive_weight is not None:
        modifier_field += edge_additive_weight[..., eid]

    return func(
        field,
        did,
        modifier_field,
    )
