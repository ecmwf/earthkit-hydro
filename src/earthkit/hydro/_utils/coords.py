def get_core_grid_dims(ds):
    possible_names = [["lat", "lon"], ["latitude", "longitude"], ["y", "x"]]
    for names in possible_names:
        present = True
        for name in names:
            present &= name in ds.coords
        if present:
            return names


def get_core_node_dims(ds):
    possible_names = [
        ["index"],
        ["node_index"],
        ["node_id"],
        ["station_index"],
        ["station_id"],
        ["gauge_id"],
        ["id"],
        ["idx"],
    ]
    for names in possible_names:
        present = True
        for name in names:
            present &= name in ds.coords
        if present:
            return names


def get_core_edge_dims(ds):
    possible_names = [["edge_id"]]
    for names in possible_names:
        present = True
        for name in names:
            present &= name in ds.coords
        if present:
            return names


def get_core_dims(ds):
    dims = get_core_grid_dims(ds)
    if dims is None:
        dims = get_core_node_dims(ds)
    if dims is None:
        dims = get_core_edge_dims(ds)
    if dims is None:
        raise ValueError("Could not autodetect xarray core dims.")
    return dims


node_default_coord = "node_index"
