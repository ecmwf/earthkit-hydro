import numpy as np
import joblib


def mask_data(func):
    def wrapper(self, field, *args, **kwargs):
        if field.shape[-2:] == self.mask.shape:
            in_place = kwargs.get("in_place", False)
            if in_place:
                return func(self, field[..., self.mask].T, *args, **kwargs).T
            else:
                out_field = field.copy()
                new_field = func(self, field[..., self.mask].T, *args, **kwargs).T
                out_field[..., self.mask] = new_field
                return out_field
        else:
            return func(self, field.T, *args, **kwargs).T

    return wrapper


class RiverNetwork:
    def __init__(self, nodes, downstream, mask) -> None:
        self.nodes = nodes
        self.n_nodes = len(nodes)
        self.downstream_nodes = downstream
        self.mask = mask
        # Note: a node with no upstream or downstream is considered both a source and a sink
        self.sinks = self.nodes[self.downstream_nodes == self.n_nodes]  # nodes with no downstreams
        print("finding sources")
        self.sources = self.get_sources()  # nodes with no upstreams
        print("topological sorting")
        self.topological_groups = self.topological_sort()

    def get_sources(self):
        tmp_nodes = self.nodes.copy()
        downstream_no_sinks = self.downstream_nodes[self.downstream_nodes != self.n_nodes]  # get all downstream nodes
        tmp_nodes[downstream_no_sinks] = -1  # downstream nodes that aren't sinks = -1
        inlets = tmp_nodes[tmp_nodes != -1]  # sources are nodes that are not downstream nodes
        return inlets

    def topological_sort(self):
        inlets = self.sources
        labels = np.zeros(self.n_nodes, dtype=int)

        # this does a BFS from the sources to find largest distance to each node
        old_sum = -1
        current_sum = 0  # sum of labels
        n = 1  # distance from source
        while (
            current_sum > old_sum
        ):  # sum is monotonic increasing since we only increase labels. Could also do a != check here (is equivalent)
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
        sorted_indices = np.argsort(labels)  # sort by labels
        sorted_array = self.nodes[sorted_indices]
        sorted_labels = labels[sorted_indices]
        _, indices = np.unique(sorted_labels, return_index=True)  # find index of first occurrence of each label
        subarrays = np.split(sorted_array, indices[1:])  # split array at each first occurrence of a label
        return subarrays

    @mask_data
    def accuflux(self, field, in_place=False, operation=np.add):
        # propagates a field all the way downstream along the river network
        if not in_place:
            field = field.copy()
        for grouping in self.topological_groups[
            :-1
        ]:  # exclude sinks here since they have nowhere to propagate downstream
            operation.at(field, self.downstream_nodes[grouping], field[grouping])
        return field

    @mask_data
    def upstream(self, field):
        # update each node with the sum of its parent (upstream) nodes
        mask = self.downstream_nodes != self.n_nodes  # remove sinks since they have no downstream
        ups = np.zeros(self.n_nodes, dtype=field.dtype)
        np.add.at(
            ups, self.downstream_nodes[mask], field[mask]
        )  # update each downstream node with the sum of its upstream nodes
        # ups[mask] = None # set sinks to have None as upstream contribution?
        return ups

    @mask_data
    def downstream(self, field):
        # update each node with its children (downstream) node (currently only one downstream node)
        down = np.zeros(self.n_nodes, dtype=field.dtype)
        mask = self.downstream_nodes != self.n_nodes  # remove sinks
        down[mask] = field[self.downstream_nodes[mask]]
        # down[~mask] = None # set sinks to have None downstream?
        return down

    @mask_data
    def catchment(self, field, overwrite=True):
        # this always loops over the entire length of the chain
        # Better would be to do a BFS starting at each of the nodes
        # but this will be slow in Python

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
        return self.catchment(field, overwrite=False)

    def export(self, fname="river_network.joblib", compress=1):
        joblib.dump(self, fname, compress=compress)
