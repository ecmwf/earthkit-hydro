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
