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
        up_ids_upsort,  # sorted by top groups up
        down_ids_upsort,  # sorted by top groups up
        edge_ids_upsort,  # sorted by top groups up
        up_ids_downsort,  # sorted by top groups down
        down_ids_downsort,  # sorted by top groups down
        edge_ids_downsort,  # sorted by top groups down
        sources,
        sinks,
        coords,
        mask,
        bifurcates,
        up_splits,  # just indices of where to split
        down_splits,  # just indices of where to split
    ):
        self.n_nodes = n_nodes
        self.n_edges = n_edges
        self.bifurcates = bifurcates
        self.sources = sources
        self.sinks = sinks
        self.coords = coords
        self.mask = mask
        self.up_ids_upsort = up_ids_upsort
        self.down_ids_upsort = down_ids_upsort
        self.edge_ids_upsort = edge_ids_upsort
        self.up_ids_downsort = up_ids_downsort
        self.down_ids_downsort = down_ids_downsort
        self.edge_ids_downsort = edge_ids_downsort
        self.up_splits = up_splits
        self.down_splits = down_splits


class RiverNetwork:

    def __init__(self, river_network_storage):
        self._storage = river_network_storage
        self.n_nodes = self._storage.n_nodes
        self.n_edges = self._storage.n_edges
        self._nodes = np.arange(self.n_nodes)  # TODO: check if needed
        self.sources = self._storage.sources
        self.sinks = self._storage.sinks

        self.downstream_groups = list(
            zip(
                np.split(self._storage.down_ids_downsort, self._storage.down_splits),
                np.split(self._storage.up_ids_downsort, self._storage.down_splits),
                np.split(self._storage.edge_ids_downsort, self._storage.down_splits),
            )
        )
        self.upstream_groups = list(
            zip(
                np.split(self._storage.up_ids_upsort, self._storage.up_splits),
                np.split(self._storage.down_ids_upsort, self._storage.up_splits),
                np.split(self._storage.edge_ids_upsort, self._storage.up_splits),
            )
        )

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
