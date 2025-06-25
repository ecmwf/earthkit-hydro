# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import joblib
import numpy as np


class RiverNetworkStorage:

    def __init__(
        self,
        n_nodes,
        n_edges,
        up_ids,
        down_ids,
        coords,
        mask,
        bifurcates,
        downstream_group_labels,
        upstream_group_labels,
    ):
        self.n_nodes = n_nodes
        self.n_edges = n_edges
        self.bifurcates = bifurcates
        del n_edges, n_nodes, bifurcates
        self.up_ids = up_ids
        del up_ids
        self.down_ids = down_ids
        del down_ids
        self.coords = coords
        del coords
        self.mask = mask
        del mask
        self.downstream_group_labels = downstream_group_labels
        del downstream_group_labels
        self.upstream_group_labels = upstream_group_labels
        del upstream_group_labels


def split_by_labels(labels, sorted_indices, array):
    sorted_array = array[sorted_indices]
    sorted_labels = labels[sorted_indices]
    _, indices = np.unique(sorted_labels, return_index=True)
    subarrays = np.split(sorted_array, indices[1:])
    return subarrays


def get_sources(down_inds, up_inds):
    return np.setdiff1d(up_inds, down_inds, assume_unique=False)


def get_sinks(down_inds, up_inds):
    return np.setdiff1d(down_inds, up_inds, assume_unique=False)


def get_singletons(down_inds, up_inds, nodes):
    connected = np.union1d(up_inds, down_inds)
    return np.setdiff1d(nodes, connected)


class RiverNetwork:

    def __init__(self, river_network_storage):
        self._storage = river_network_storage
        self.n_nodes = self._storage.n_nodes
        self.n_edges = self._storage.n_edges

        self._edges = np.arange(self.n_edges)
        self._nodes = np.arange(self.n_nodes)

        self.singletons = get_singletons(
            self._storage.down_ids, self._storage.up_ids, self._nodes
        )
        self.sources = get_sources(self._storage.down_ids, self._storage.up_ids)
        self.sinks = get_sinks(self._storage.down_ids, self._storage.up_ids)

        _sorted_indices_downstream = np.argsort(self._storage.downstream_group_labels)
        self.downstream_groups = zip(
            split_by_labels(
                self._storage.downstream_group_labels,
                _sorted_indices_downstream,
                self._storage.up_ids,
            ),
            split_by_labels(
                self._storage.downstream_group_labels,
                _sorted_indices_downstream,
                self._storage.down_ids,
            ),
            split_by_labels(
                self._storage.downstream_group_labels,
                _sorted_indices_downstream,
                self._edges,
            ),
        )
        del _sorted_indices_downstream
        _sorted_indices_upstream = np.argsort(self._storage.upstream_group_labels)
        self.upstream_groups = zip(
            split_by_labels(
                self._storage.upstream_group_labels,
                _sorted_indices_upstream,
                self._storage.down_ids,
            ),
            split_by_labels(
                self._storage.upstream_group_labels,
                _sorted_indices_upstream,
                self._storage.up_ids,
            ),
            split_by_labels(
                self._storage.upstream_group_labels,
                _sorted_indices_upstream,
                self._edges,
            ),
        )
        del _sorted_indices_upstream

    @property
    def mask(self):
        if self._storage.mask is None:
            raise ValueError(
                "This RiverNetwork is not raster-based and does not have a mask."
            )
        return self._storage.mask

    @property
    def shape(self):
        return self.mask.shape

    def __str__(self):
        return f"RiverNetwork with {self.n_nodes} nodes and {self.n_edges} edges."

    def __repr__(self):
        return self.__str__()

    def to(self, backend="numpy", dev=None):
        raise NotImplementedError(
            f"Switching array backend to {backend} on device {dev} not yet supported."
        )

    def export(self, fpath="river_network.joblib", compression=1):
        joblib.dump(self._storage, fpath, compress=compression)

    def create_subnetwork(self, *args, **kwargs):
        raise NotImplementedError("Subnetwork creation not yet supported.")

    # @mask_2d
    # def create_subnetwork(self, node_mask=None, edge_mask=None, recompute=True):
    #     """Creates a subnetwork from the river network based on a mask.

    #     Parameters
    #     ----------
    #     field : numpy.ndarray
    #         A boolean mask to subset the river network.
    #     recompute : bool, optional
    #         If True, recomputes the topological labels for the subnetwork.
    #         Default is False.

    #     Returns
    #     -------
    #     RiverNetwork
    #         A subnetwork of the river network.

    #     """
    #     assert not (node_mask is None and edge_mask is None)
    #     if self.has_bifurcations:
    #         raise NotImplementedError("Bifurcations not yet supported.")
    #     if edge_mask is not None:
    #         raise NotImplementedError("edge_mask not yet supported.")
    #     domain_mask, river_network_mask = _find_new_masks(self.mask, node_mask)

    #     nodes, downstream, n_nodes, sinks, sources = _find_subnetwork_inputs(
    #         river_network_mask, self.downstream_nodes, self.n_nodes
    #     )
    #     topological_labels = self.topological_labels[river_network_mask]
    #     del river_network_mask
    #     topological_labels[sources] = 0
    #     topological_labels[sinks] = self.n_nodes
    #     network = RiverNetwork(
    #         nodes,
    #         np.where(domain_mask),
    #         domain_mask.shape,
    #         downstream,
    #         sinks=sinks,
    #         sources=sources,
    #         n_nodes=n_nodes,
    #         topological_labels=topological_labels,
    #         mask=domain_mask,
    #     )
    #     del nodes, downstream, domain_mask, sinks, sources, n_nodes
    #     del topological_labels
    #     if recompute:
    #         topological_labels = distance_to_source(network, path="longest")[
    #             network.mask
    #         ].astype(int)
    #         topological_labels[network.sinks] = network.n_nodes
    #         network.topological_labels = topological_labels
    #         del topological_labels
    #         topological_groups = network.topological_groups_from_labels()
    #         network.topological_groups_edges = []
    #         for grouping in topological_groups[:-1]:
    #             network.topological_groups_edges.append(grouping)
    #     return network
