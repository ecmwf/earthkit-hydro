# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import joblib
import numpy as np

from .distance import to_source as distance_to_source
from .utils import mask_2d


class RiverNetwork:
    """A class representing a river network for hydrological processing.

    Attributes
    ----------


    """

    def __init__(
        self,
        nodes,
        grid_ids,
        shape,
        down_ids,
        up_ids=None,
        has_bifurcations=False,
        mask=None,
        sinks=None,
        sources=None,
        n_nodes=None,
        topological_labels=None,
    ) -> None:
        """Initialises the RiverNetwork with nodes, downstream nodes, and a
        mask.

        Parameters
        ----------

        """
        assert (up_ids is None) == (has_bifurcations is False)

        self.nodes = nodes
        del nodes
        self.n_nodes = len(self.nodes) if n_nodes is None else n_nodes
        self._mask = mask
        del mask
        self.shape = shape if shape is not None else self.mask.shape
        del shape
        self.grid_ids = grid_ids
        del grid_ids
        self.has_bifurcations = has_bifurcations
        del has_bifurcations

        if not self.has_bifurcations:
            self.downstream_nodes = down_ids
            del down_ids

            self.sinks = (
                sinks
                if sinks is not None
                else self.nodes[self.downstream_nodes == self.n_nodes]
            )  # nodes with no downstreams
            del sinks
            self.sources = (
                sources
                if sources is not None
                else _get_sources(self.nodes, self.downstream_nodes, self.n_nodes)
            )  # nodes with no upstreams
            del sources

            self.topological_labels = (
                topological_labels
                if topological_labels is not None
                else self.compute_topological_labels_no_bifurcations()
            )
            del topological_labels
            topological_groups = self.topological_groups_from_labels()
            self.topological_groups_edges = []
            for grouping in topological_groups[:-1]:
                self.topological_groups_edges.append(grouping)
            self.get_up_down = self.get_up_down_no_bifurcations
        else:
            counts = np.bincount(up_ids, minlength=self.n_nodes)
            offsets = np.zeros(self.n_nodes + 1, dtype=int)
            offsets[1:] = np.cumsum(counts)
            del counts
            has_incoming = np.isin(self.nodes, down_ids)
            self.sources = self.nodes[~has_incoming]
            del has_incoming
            self.sinks = np.where(offsets[1:] == offsets[:-1])[0]
            from .network_utils import (
                compute_topological_labels_bifurcations,
                get_edge_indices_numba,
            )

            self.topological_labels = compute_topological_labels_bifurcations(
                down_ids, offsets, self.sources, self.sinks
            )
            del topological_labels
            topological_groups = self.topological_groups_from_labels()
            self.up_ids = up_ids
            self.down_ids = down_ids
            self.topological_groups_edges = []
            for grouping in topological_groups[:-1]:
                edges = get_edge_indices_numba(offsets, grouping)
                self.topological_groups_edges.append(edges)
            self.get_up_down = self.get_up_down_bifurcations

    def get_up_down_bifurcations(self, grouping):
        return self.up_ids[grouping], self.down_ids[grouping]

    def get_up_down_no_bifurcations(self, grouping):
        return grouping, self.downstream_nodes[grouping]

    @property
    def mask(self):
        if self._mask is None:
            raise ValueError(
                "This RiverNetwork is not raster-based and does not have a mask."
            )
        return self._mask

    def compute_topological_labels_no_bifurcations(self):
        """Finds the topological distance labels for each node in the river
        network.

        Returns
        -------
        numpy.ndarray
            Array of topological distance labels for each node.

        """
        try:
            from .topological_labels_rust import compute_topological_labels
        except (ModuleNotFoundError, ImportError):
            print(
                "Failed to load rust extension, falling back to python implementation."
            )
            from .topological_labels_python import compute_topological_labels
        return compute_topological_labels(
            self.sources, self.sinks, self.downstream_nodes
        )

    def topological_groups_from_labels(self):
        """Groups nodes by their topological distance labels.

        Parameters
        ----------
        labels : numpy.ndarray
            Array of labels representing the topological distances of nodes.

        Returns
        -------
        list of numpy.ndarray
            A list of subarrays, each containing nodes with the same label.

        """
        sorted_indices = np.argsort(self.topological_labels)  # sort by labels
        sorted_array = self.nodes[sorted_indices]
        sorted_labels = self.topological_labels[sorted_indices]
        _, indices = np.unique(
            sorted_labels, return_index=True
        )  # find index of first occurrence of each label
        subarrays = np.split(
            sorted_array, indices[1:]
        )  # split array at each first occurrence of a label
        return subarrays

    def export(self, fpath="river_network.joblib", compression=1):
        """Exports the river network instance to a file.

        Parameters
        ----------
        fpath : str, optional
            The filepath to save the instance (default is "river_network.joblib").
        compression : int, optional
            Compression level for joblib (default is 1).

        """
        joblib.dump(self, fpath, compress=compression)

    def __str__(self):
        """Returns a string representation of the river network."""
        return f""""
            RiverNetwork with {self.n_nodes} nodes,
            defined on a {self.mask.shape[0]}x{self.mask.shape[1]} grid.
            """

    def __repr__(self):
        """Returns a string representation of the river network."""
        return self.__str__()

    @mask_2d
    def create_subnetwork(self, mask, recompute=True):
        """Creates a subnetwork from the river network based on a mask.

        Parameters
        ----------
        field : numpy.ndarray
            A boolean mask to subset the river network.
        recompute : bool, optional
            If True, recomputes the topological labels for the subnetwork.
            Default is False.

        Returns
        -------
        RiverNetwork
            A subnetwork of the river network.

        """

        assert not self.has_bifurcations
        domain_mask, river_network_mask = _find_new_masks(self.mask, mask)

        nodes, downstream, n_nodes, sinks, sources = _find_subnetwork_inputs(
            river_network_mask, self.downstream_nodes, self.n_nodes
        )
        topological_labels = self.topological_labels[river_network_mask]
        del river_network_mask
        topological_labels[sources] = 0
        topological_labels[sinks] = self.n_nodes
        network = RiverNetwork(
            nodes,
            np.where(domain_mask),
            domain_mask.shape,
            downstream,
            sinks=sinks,
            sources=sources,
            n_nodes=n_nodes,
            topological_labels=topological_labels,
            mask=domain_mask,
        )
        del nodes, downstream, domain_mask, sinks, sources, n_nodes, topological_labels
        if recompute:
            topological_labels = distance_to_source(network, path="longest")[
                network.mask
            ].astype(int)
            topological_labels[network.sinks] = network.n_nodes
            network.topological_labels = topological_labels
            del topological_labels
            topological_groups = network.topological_groups_from_labels()
            network.topological_groups_edges = []
            for grouping in topological_groups[:-1]:
                network.topological_groups_edges.append(
                    (grouping, network.downstream_nodes[grouping])
                )
        return network


def _get_sources(nodes, downstream_nodes, n_nodes):
    """Identifies the source nodes in the river network (nodes with no
    upstream nodes).

    Returns
    -------
    numpy.ndarray
        Array of source nodes.

    """
    tmp_nodes = nodes.copy()
    downstream_no_sinks = downstream_nodes[
        downstream_nodes != n_nodes
    ]  # get all downstream nodes
    tmp_nodes[downstream_no_sinks] = (
        n_nodes + 1
    )  # downstream nodes that aren't sinks = -1
    inlets = tmp_nodes[
        tmp_nodes != n_nodes + 1
    ]  # sources are nodes that are not downstream nodes
    return inlets


def _find_new_masks(original_mask, mask):
    if mask.ndim == 1:
        river_network_mask = mask
        valid_indices = np.where(original_mask)
        new_valid_indices = (
            valid_indices[0][river_network_mask],
            valid_indices[1][river_network_mask],
        )
        domain_mask = np.full(original_mask.shape, False)
        domain_mask[new_valid_indices] = True
    else:
        domain_mask = mask & original_mask
        river_network_mask = domain_mask[original_mask]

    return domain_mask, river_network_mask


def _find_subnetwork_inputs(
    river_network_mask, original_downstream_nodes, original_n_nodes
):
    downstream_indices = original_downstream_nodes[river_network_mask]
    n_nodes = len(downstream_indices)  # number of nodes in the subnetwork
    # create new array of network nodes, setting all nodes not in mask to n_nodes
    subnetwork_nodes = np.full(original_n_nodes, n_nodes)
    subnetwork_nodes[river_network_mask] = np.arange(n_nodes)
    # get downstream nodes in the subnetwork
    non_sinks = downstream_indices != original_n_nodes
    downstream = np.full(n_nodes, n_nodes, dtype=np.uintp)
    downstream[non_sinks] = subnetwork_nodes[downstream_indices[non_sinks]]
    nodes = np.arange(n_nodes, dtype=np.uintp)

    sinks = nodes[downstream == n_nodes]
    sources = _get_sources(nodes, downstream, n_nodes)

    return nodes, downstream, n_nodes, sinks, sources
