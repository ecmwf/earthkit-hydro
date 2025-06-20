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
from .network_utils import (
    _find_new_masks,
    _find_subnetwork_inputs,
    compute_topological_labels_bifurcations,
    get_offsets,
    get_sinks_bifurcations,
    get_sinks_no_bifurcations,
    get_sources_bifurcations,
    get_sources_no_bifurcations,
)
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
        self.n_nodes = self.get_n_nodes(n_nodes)
        self._mask = mask
        del mask
        self.shape = self.get_shape(shape)
        del shape
        self.grid_ids = grid_ids
        del grid_ids
        self.has_bifurcations = has_bifurcations
        del has_bifurcations

        if not self.has_bifurcations:
            self.downstream_nodes = down_ids
            del down_ids
            self.sources = get_sources_no_bifurcations(
                sources, self.nodes, self.downstream_nodes, self.n_nodes
            )
            del sources
            self.sinks = get_sinks_no_bifurcations(
                sinks, self.nodes, self.downstream_nodes, self.n_nodes
            )
            del sinks
            self.topological_labels = (
                topological_labels
                if topological_labels is not None
                else self.compute_topological_labels_no_bifurcations()
            )
            del topological_labels
            self.get_up_down = self.get_up_down_no_bifurcations

            def get_group(grouping):
                return grouping

            self.n_edges = self.n_nodes - len(self.sinks)

        else:
            self.up_ids = up_ids
            del up_ids
            self.down_ids = down_ids
            del down_ids
            offsets = get_offsets(self.up_ids, self.n_nodes)
            self.sources = get_sources_bifurcations(sources, self.nodes, self.down_ids)
            del sources
            self.sinks = get_sinks_bifurcations(sinks, offsets)
            del sinks
            self.topological_labels = (
                topological_labels
                if topological_labels is not None
                else compute_topological_labels_bifurcations(
                    self.down_ids, offsets, self.sources, self.sinks
                )
            )
            del topological_labels
            self.get_up_down = self.get_up_down_bifurcations
            from ._numba import get_edge_indices_numba

            def get_group(grouping):
                return get_edge_indices_numba(offsets, grouping)

            self.n_edges = self.down_ids.shape

        topological_groups = self.topological_groups_from_labels()
        self.topological_groups_edges = []
        for grouping in topological_groups[:-1]:
            edges = get_group(grouping)
            self.topological_groups_edges.append(edges)

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

    def get_shape(self, shape):
        return shape if shape is not None else self.mask.shape

    def get_n_nodes(self, n_nodes):
        return n_nodes if n_nodes is not None else len(self.nodes)

    def __str__(self):
        """Returns a string representation of the river network."""
        return f""""
            RiverNetwork with {self.n_nodes} nodes,
            defined on a {self.shape[0]}x{self.shape[1]} grid.
            """

    def __repr__(self):
        """Returns a string representation of the river network."""
        return self.__str__()

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

    @mask_2d
    def create_subnetwork(self, node_mask=None, edge_mask=None, recompute=True):
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
        assert not (node_mask is None and edge_mask is None)
        if self.has_bifurcations:
            raise NotImplementedError("Bifurcations not yet supported.")
        if edge_mask is not None:
            raise NotImplementedError("edge_mask not yet supported.")
        domain_mask, river_network_mask = _find_new_masks(self.mask, node_mask)

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
                network.topological_groups_edges.append(grouping)
        return network
