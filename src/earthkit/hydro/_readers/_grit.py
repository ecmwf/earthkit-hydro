import numpy as np

from earthkit.hydro.data_structures._network_storage import RiverNetworkStorage

from ._core import get_sources


def get_edge_indices(offsets, grouping):
    lengths = offsets[grouping + 1] - offsets[grouping]
    total_len = np.sum(lengths)
    result = np.empty(total_len, dtype=int)
    pos = 0

    for node, length in zip(grouping, lengths):
        start = offsets[node]
        for j in range(length):
            result[pos + j] = start + j
        pos += length
    return result


def compute_topological_labels_bifurcations(down_ids, offsets, sources, sinks):
    n_nodes = offsets.size - 1
    labels = np.zeros(n_nodes, dtype=int)
    inlets = sources

    for n in range(1, n_nodes + 1):
        inlets = np.unique(down_ids[get_edge_indices(offsets, inlets)])
        if inlets.size == 0:
            labels[sinks] = n - 1
            break
        labels[inlets] = n

    return labels


def from_grit(path):
    import geopandas as gpd

    nodes_df = gpd.read_file(path, layer="nodes")
    lines_df = gpd.read_file(path, layer="lines")

    try:
        nodes_df["x"] = nodes_df.geometry.x
        nodes_df["y"] = nodes_df.geometry.y
    except Exception:
        nodes_df["geometry"] = nodes_df["geometry"].apply(lambda geom: geom.geoms[0])
        nodes_df["x"] = nodes_df.geometry.x
        nodes_df["y"] = nodes_df.geometry.y

    nodes_df.sort_values(by=["y", "x"], inplace=True, ascending=[False, True])
    nodes_df.reset_index(inplace=True)

    ref = nodes_df["global_id"]

    value_to_index = dict(zip(ref.values, ref.index.values))
    lines_df["UPID"] = lines_df["upstream_node_id"].map(value_to_index)
    lines_df["DOWNID"] = lines_df["downstream_node_id"].map(value_to_index)
    lines_df.sort_values(by=["UPID", "DOWNID"], inplace=True)
    up_ids = lines_df["UPID"].to_numpy()
    down_ids = lines_df["DOWNID"].to_numpy()
    edge_weights = lines_df["width_adjusted"].to_numpy()
    np.nan_to_num(edge_weights, copy=False, nan=1)

    shape = None
    n_nodes = nodes_df.shape[0]
    n_edges = lines_df.shape[0]
    pixarea = None
    bifurcates = True
    mask = None
    coords = {"y": nodes_df["y"].to_numpy(), "x": nodes_df["x"].to_numpy()}

    del nodes_df, lines_df

    sources = get_sources(n_nodes, down_ids)
    sinks = get_sources(n_nodes, up_ids)

    assert np.all(np.isin(np.setdiff1d(sinks, sources), down_ids))

    counts = np.bincount(up_ids, minlength=n_nodes)
    offsets = np.zeros(n_nodes + 1, dtype=int)
    offsets[1:] = np.cumsum(counts)
    del counts

    topological_labels = compute_topological_labels_bifurcations(
        down_ids, offsets, sources, sinks
    )
    topological_labels = topological_labels[up_ids]

    sort_indices = np.argsort(topological_labels)
    sorted_distances = topological_labels[sort_indices]  # from source to sink

    edge_indices = np.arange(n_edges)

    up_ids_sort = up_ids[sort_indices]
    down_ids_sort = down_ids[sort_indices]
    edge_ids_sort = edge_indices[sort_indices]

    _, splits = np.unique(sorted_distances, return_index=True)
    splits = splits[1:]

    edge_weights_per_node = np.zeros(n_nodes)
    np.add.at(edge_weights_per_node, up_ids_sort, edge_weights[sort_indices])
    edge_weights_norm = np.empty(n_edges)
    edge_weights_norm[edge_ids_sort] = edge_weights_per_node[up_ids_sort]
    del edge_weights_per_node
    edge_weights /= edge_weights_norm
    del edge_weights_norm

    store = RiverNetworkStorage(
        n_nodes,
        n_edges,
        np.vstack([down_ids_sort, up_ids_sort, edge_ids_sort]).astype(np.int64),
        sources,
        sinks,
        coords,
        splits,
        pixarea,
        mask,
        shape,
        bifurcates,
        edge_weights,
    )

    return store
