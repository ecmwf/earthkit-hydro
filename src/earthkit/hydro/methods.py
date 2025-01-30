import numpy as np
from .river_network import RiverNetwork


def is_missing(field, mv):
    """
    Finds a mask of missing values.

    Parameters
    ----------
    field : numpy.ndarray
        The scalar input field to check for missing values.
    mv : scalar
        The missing value to check for.

    Returns
    -------
    numpy.ndarray
        A boolean mask of missing values.
    """
    if np.isnan(mv):
        return np.isnan(field)
    elif np.isinf(mv):
        return np.isinf(field)
    else:
        return field == mv


def are_missing_values_present(field, mv):
    """
    Finds if missing values are present in a field.

    Parameters
    ----------
    field : numpy.ndarray
        The scalar input field to check for missing values.
    mv : scalar
        The missing value to check for.

    Returns
    -------
    bool
        True if missing values are present, False otherwise.
    """
    return np.any(is_missing(field, mv))


def check_missing(field, mv, accept_missing):
    """
    Finds missing values and checks if they are allowed in the input field.

    Parameters
    ----------
    field : numpy.ndarray
        The scalar input field to check for missing values.
    mv : scalar
        The missing value to check for.
    accept_missing : bool
        If True, missing values are allowed in the input field.

    Returns
    -------
    bool
        True if missing values are present, False otherwise.
    """
    missing_values_present = are_missing_values_present(field, mv)
    if missing_values_present:
        if not accept_missing:
            raise ValueError("Missing values present in input field and accept_missing is False.")
        else:
            print("Warning: missing values present in input field.")
    return missing_values_present


def mask_2d(func):
    """
    Decorator to allow function to mask 2d inputs to the river network.

    Parameters
    ----------
    func : callable
        The function to be wrapped and executed with masking applied.

    Returns
    -------
    callable
        The wrapped function.
    """

    def wrapper(river_network, field, *args, **kwargs):
        """
        Wrapper masking 2d data fields to allow for processing along the river network, then undoing the masking.

        Parameters
        ----------
        river_network : object
            The RiverNetwork instance calling the method.
        field : numpy.ndarray
            The input data field to be processed.
        *args : tuple
            Positional arguments passed to the wrapped function.
        **kwargs : dict
            Keyword arguments passed to the wrapped function.

        Returns
        -------
        numpy.ndarray
            The processed field.
        """
        if field.shape[-2:] == river_network.mask.shape:
            return func(river_network, field[..., river_network.mask].T, *args, **kwargs)
        else:
            return func(river_network, field.T, *args, **kwargs)

    return wrapper


def mask_and_unmask_data(func):
    """
    Decorator to convert masked 2d inputs back to 1d.

    Parameters
    ----------
    func : callable
        The function to be wrapped and executed with masking applied.

    Returns
    -------
    callable
        The wrapped function.
    """

    def wrapper(river_network, field, *args, **kwargs):
        """
        Wrapper masking 2d data fields to allow for processing along the river network, then undoing the masking.

        Parameters
        ----------
        river_network : object
            The RiverNetwork instance calling the method.
        field : numpy.ndarray
            The input data field to be processed.
        *args : tuple
            Positional arguments passed to the wrapped function.
        **kwargs : dict
            Keyword arguments passed to the wrapped function.

        Returns
        -------
        numpy.ndarray
            The processed field.
        """
        # gets the missing value from the keyword arguments if it is present, otherwise takes default value of mv from func
        mv = kwargs.get("mv")
        mv = mv if mv is not None else func.__defaults__[0]
        if field.shape[-2:] == river_network.mask.shape:
            in_place = kwargs.get("in_place", False)
            if in_place:
                out_field = field
            else:
                out_field = np.empty(field.shape, dtype=field.dtype)
            out_field[..., river_network.mask] = func(
                river_network, field[..., river_network.mask].T, *args, **kwargs
            ).T

            out_field[..., ~river_network.mask] = mv
            return out_field
        else:
            return func(river_network, field.T, *args, **kwargs).T

    return wrapper


@mask_and_unmask_data
def accuflux(river_network, field, mv=np.nan, in_place=False, operation=np.add, accept_missing=False):
    """
    Accumulate a field downstream along the river network.

    Parameters
    ----------
    field : numpy.ndarray
        The input field to propagate.
    mv : scalar, optional
        The missing value to use (default is np.nan).
    in_place : bool, optional
        If True, modifies the field in-place (default is False).
    operation : callable, optional
        The operation to perform when propagating (default is numpy.add).
    accept_missing : bool, optional
        If True, missing values are allowed in the input field (default is False).

    Returns
    -------
    numpy.ndarray
        The propagated field.
    """

    missing_values_present = check_missing(field, mv, accept_missing)

    if not in_place:
        field = field.copy()

    if not missing_values_present:
        for grouping in river_network.topological_groups[:-1]:
            operation.at(field, river_network.downstream_nodes[grouping], field[grouping])
    else:
        for grouping in river_network.topological_groups[:-1]:
            nodes_to_update = river_network.downstream_nodes[grouping]
            values_to_add = field[grouping]
            missing_indices = np.logical_or(is_missing(field[grouping], mv), is_missing(field[nodes_to_update], mv))
            operation.at(field, nodes_to_update, values_to_add)
            field[nodes_to_update[missing_indices]] = mv
    return field


@mask_and_unmask_data
def upstream(river_network, field, mv=np.nan, operation=np.add, accept_missing=False):
    """
    Sets each node to be the sum of its upstream nodes values, or a missing value.

    Parameters
    ----------
    field : numpy.ndarray
        The input field representing node values.
    mv : scalar, optional
        The missing value to use (default is np.nan).
    operation : callable, optional
        The operation to perform when propagating (default is numpy.add).
    accept_missing : bool, optional
        If True, missing values are allowed in the input field (default is False

    Returns
    -------
    numpy.ndarray
        The updated field with upstream contributions.
    """
    missing_values_present = check_missing(field, mv, accept_missing)

    ups = np.zeros(river_network.n_nodes, dtype=field.dtype)
    mask = river_network.downstream_nodes != river_network.n_nodes  # remove sinks since they have no downstream
    nodes_to_update = river_network.downstream_nodes[mask]
    values_to_add = field[mask]
    operation.at(ups, nodes_to_update, values_to_add)
    if missing_values_present:
        missing_indices = is_missing(values_to_add, mv)
        ups[nodes_to_update[missing_indices]] = mv
    return ups


@mask_and_unmask_data
def downstream(river_network, field, mv=np.nan, accept_missing=False):
    """
    Sets each node to be its downstream node value, or a missing value.

    Parameters
    ----------
    field : numpy.ndarray
        The input field representing node values.
    mv : scalar, optional
        The missing value to use (default is np.nan).
    accept_missing : bool, optional
        If True, missing values are allowed in the input field (default is False).

    Returns
    -------
    numpy.ndarray
        The updated field with downstream values.
    """
    _ = check_missing(field, mv, accept_missing)

    down = np.zeros(river_network.n_nodes, dtype=field.dtype)
    mask = river_network.downstream_nodes != river_network.n_nodes  # remove sinks
    down[mask] = field[river_network.downstream_nodes[mask]]
    return down


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


@mask_and_unmask_data
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
    return river_network.catchment(field, mv=mv, overwrite=False)


@mask_2d
def create_subnetwork(river_network, field, recompute=False):
    """
    Creates a subnetwork from the river network based on a mask.

    Parameters
    ----------
    field : numpy.ndarray
        A boolean mask to subset the river network.
    recompute : bool, optional
        If True, recomputes the topological labels for the subnetwork (default is False).

    Returns
    -------
    RiverNetwork
        A subnetwork of the river network.
    """
    river_network_mask = field
    valid_indices = np.where(river_network.mask)
    new_valid_indices = (valid_indices[0][river_network_mask], valid_indices[1][river_network_mask])
    domain_mask = np.full(river_network.mask.shape, False)
    domain_mask[new_valid_indices] = True

    downstream_indices = river_network.downstream_nodes[river_network_mask]
    n_nodes = len(downstream_indices)  # number of nodes in the subnetwork
    # create new array of network nodes, setting all nodes not in mask to n_nodes
    subnetwork_nodes = np.full(river_network.n_nodes, n_nodes)
    subnetwork_nodes[river_network_mask] = np.arange(n_nodes)
    # get downstream nodes in the subnetwork
    non_sinks = np.where(downstream_indices != river_network.n_nodes)
    downstream = np.full(n_nodes, n_nodes)
    downstream[non_sinks] = subnetwork_nodes[downstream_indices[non_sinks]]
    nodes = np.arange(n_nodes)

    if not recompute:
        sinks = nodes[downstream == n_nodes]
        topological_labels = river_network.topological_labels[river_network_mask]
        topological_labels[sinks] = river_network.n_nodes

        return RiverNetwork(nodes, downstream, domain_mask, sinks=sinks, topological_labels=topological_labels)
    else:
        return RiverNetwork(nodes, downstream, domain_mask)
