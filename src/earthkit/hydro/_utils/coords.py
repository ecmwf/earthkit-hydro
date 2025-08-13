def get_core_grid_dims(ds):
    possible_names = [["lat", "lon"], ["latitude", "longitude"], ["y", "x"]]
    for names in possible_names:
        present = True
        for name in names:
            present &= name in ds.coords
        if present:
            return names

    return None


def get_core_1d_dims(ds):
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

    return None
