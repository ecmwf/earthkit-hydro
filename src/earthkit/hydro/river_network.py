import numpy as np
import joblib


def get_missing_value(field, mv=None):
    """
    Find the missing value for a field.

    Parameters
    ----------
    field: numpy.ndarray
        The input field to find the missing value for.

    Returns
    -------
    float or int
        The missing value for the field
    """

    if mv is None:
        if issubclass(field.dtype.type, np.floating):
            return np.nan
        elif issubclass(field.dtype.type, np.integer):
            return 0
        else:
            raise Exception("Input field is neither float nor integer type")
    else:
        if issubclass(field.dtype.type, np.floating):
            return mv
        elif issubclass(field.dtype.type, np.integer):
            if np.isfinite(mv):
                return int(mv)
            else:
                raise Exception("Integer field cannot accept non-finite missing values")
        else:
            raise Exception("Input field is neither float nor integer type")


def check_no_missing(field, mv, accept_missing):
    mv = get_missing_value(field, mv=mv)
    if not accept_missing and np.any(field == mv if not np.isnan(mv) else np.isnan(field)):
        raise Exception("Field contains missing values.")


def mask_data(func):
    """
    Decorator to allow function to accept 2d inputs.

    Parameters
    ----------
    func : callable
        The function to be wrapped and executed with masking applied.

    Returns
    -------
    callable
        The wrapped function.
    """

    def wrapper(self, field, *args, **kwargs):
        """
        Wrapper masking 2d data fields to allow for processing along the river network, then undoing the masking.

        Parameters
        ----------
        self : object
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
        if field.shape[-2:] == self.mask.shape:
            in_place = kwargs.get("in_place", False)
            if in_place:
                out_field = field
            else:
                out_field = np.empty(field.shape, dtype=field.dtype)

            mv = get_missing_value(field, mv=kwargs.get("mv"))
            print(mv)

            out_field[..., self.mask] = func(self, field[..., self.mask].T, *args, **kwargs).T
            out_field[..., ~self.mask] = mv
            return out_field
        else:
            return func(self, field.T, *args, **kwargs).T

    return wrapper


class RiverNetwork:
    """
    A class representing a river network for hydrological processing.

    Attributes
    ----------
    nodes : numpy.ndarray
        Array containing the node ids of the river network.
    n_nodes : int
        The number of nodes in the river network.
    downstream_nodes : numpy.ndarray
        Array of downstream node ids corresponding to each node.
    mask : numpy.ndarray
        A mask converting from the domain to the river network.
    sinks : numpy.ndarray
        Nodes with no downstream connections.
    sources : numpy.ndarray
        Nodes with no upstream connections.
    topological_groups : list of numpy.ndarray
        Groups of nodes sorted in topological order.
    """

    def __init__(self, nodes, downstream, mask) -> None:
        """
        Initialises the RiverNetwork with nodes, downstream nodes, and a mask.

        Parameters
        ----------
        nodes : numpy.ndarray
            Array containing the node ids of the river network.
        downstream : numpy.ndarray
            Array of downstream node ids corresponding to each node.
        mask : numpy.ndarray
            A mask converting from the domain to the river network.
        """
        self.nodes = nodes
        self.n_nodes = len(nodes)
        self.downstream_nodes = downstream
        self.mask = mask
        self.sinks = self.nodes[self.downstream_nodes == self.n_nodes]  # nodes with no downstreams
        print("finding sources")
        self.sources = self.get_sources()  # nodes with no upstreams
        print("topological sorting")
        self.topological_groups = self.topological_sort()

    def get_sources(self):
        """
        Identifies the source nodes in the river network (nodes with no upstream nodes).

        Returns
        -------
        numpy.ndarray
            Array of source nodes.
        """
        tmp_nodes = self.nodes.copy()
        downstream_no_sinks = self.downstream_nodes[self.downstream_nodes != self.n_nodes]  # get all downstream nodes
        tmp_nodes[downstream_no_sinks] = -1  # downstream nodes that aren't sinks = -1
        inlets = tmp_nodes[tmp_nodes != -1]  # sources are nodes that are not downstream nodes
        return inlets

    def topological_sort(self):
        """
        Performs a topological sorting of the nodes in the river network.

        Returns
        -------
        list of numpy.ndarray
            A list of groups of nodes sorted in topological order.
        """
        inlets = self.sources
        labels = np.zeros(self.n_nodes, dtype=int)
        old_sum = -1
        current_sum = 0  # sum of labels
        n = 1  # distance from source
        while current_sum > old_sum:
            old_sum = current_sum
            inlets = inlets[inlets != self.n_nodes]  # subset to valid nodes
            labels[inlets] = n  # update furthest distance from source
            inlets = self.downstream_nodes[inlets]
            n += 1
            current_sum = np.sum(labels)
            if n > 10000:
                raise Exception("maximum iterations reached")
        labels[self.sinks] = n  # put all sinks in last group in topological ordering
        groups = self.group_labels(labels)
        return groups

    def group_labels(self, labels):
        """
        Groups nodes by their topological distance labels.

        Parameters
        ----------
        labels : numpy.ndarray
            Array of labels representing the topological distances of nodes.

        Returns
        -------
        list of numpy.ndarray
            A list of subarrays, each containing nodes with the same label.
        """
        sorted_indices = np.argsort(labels)  # sort by labels
        sorted_array = self.nodes[sorted_indices]
        sorted_labels = labels[sorted_indices]
        _, indices = np.unique(sorted_labels, return_index=True)  # find index of first occurrence of each label
        subarrays = np.split(sorted_array, indices[1:])  # split array at each first occurrence of a label
        return subarrays

    @mask_data
    def accuflux(self, field, in_place=False, operation=np.add, accept_missing=False, mv=None):
        """
        Accumulate a field downstream along the river network.

        Parameters
        ----------
        field : numpy.ndarray
            The input field to propagate.
        in_place : bool, optional
            If True, modifies the field in-place (default is False).
        operation : callable, optional
            The operation to perform when propagating (default is numpy.add).

        Returns
        -------
        numpy.ndarray
            The propagated field.
        """
        check_no_missing(field, mv, accept_missing)

        if not in_place:
            field = field.copy()
        for grouping in self.topological_groups[:-1]:
            operation.at(field, self.downstream_nodes[grouping], field[grouping])
        return field

    @mask_data
    def upstream(self, field, operation=np.add, accept_missing=False, mv=None):
        """
        Sets each node to be the sum of its upstream nodes values, or a missing value.

        Parameters
        ----------
        field : numpy.ndarray
            The input field representing node values.

        Returns
        -------
        numpy.ndarray
            The updated field with upstream contributions.
        """
        check_no_missing(field, mv, accept_missing)

        mask = self.downstream_nodes != self.n_nodes  # remove sinks since they have no downstream
        ups = np.zeros(self.n_nodes, dtype=field.dtype)
        operation.at(ups, self.downstream_nodes[mask], field[mask])
        return ups

    @mask_data
    def downstream(self, field, accept_missing=False, mv=None):
        """
        Sets each node to be its downstream node value, or a missing value.

        Parameters
        ----------
        field : numpy.ndarray
            The input field representing node values.

        Returns
        -------
        numpy.ndarray
            The updated field with downstream values.
        """
        check_no_missing(field, mv, accept_missing)

        down = np.zeros(self.n_nodes, dtype=field.dtype)
        mask = self.downstream_nodes != self.n_nodes  # remove sinks
        down[mask] = field[self.downstream_nodes[mask]]
        return down

    @mask_data
    def catchment(self, field, overwrite=True):
        """
        Propagates a field upstream to find catchments.

        Parameters
        ----------
        field : numpy.ndarray
            The input field to propagate.
        overwrite : bool, optional
            If True, overwrites existing values (default is True).

        Returns
        -------
        numpy.ndarray
            The catchment field.
        """
        if not issubclass(field.dtype.type, np.integer):
            print(field.dtype)
            raise Exception("Field is not of integer type.")

        for group in self.topological_groups[:-1][::-1]:  # exclude sinks and invert topological ordering
            valid_group = group[
                field[self.downstream_nodes[group]] != 0
            ]  # only update nodes where the downstream belongs to a catchment
            if not overwrite:
                valid_group = valid_group[field[valid_group] == 0]
            field[valid_group] = field[self.downstream_nodes[valid_group]]
        return field

    @mask_data
    def subcatchment(self, field):
        """
        Propagates a field upstream to find subcatchments.

        Parameters
        ----------
        field : numpy.ndarray
            The input field to propagate.

        Returns
        -------
        numpy.ndarray
            The propagated subcatchment field.
        """
        return self.catchment(field, overwrite=False)

    def export(self, fname="river_network.joblib", compress=1):
        """
        Exports the river network instance to a file.

        Parameters
        ----------
        fname : str, optional
            The file name to save the instance (default is "river_network.joblib").
        compress : int, optional
            Compression level for joblib (default is 1).
        """
        joblib.dump(self, fname, compress=compress)
