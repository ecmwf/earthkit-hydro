import numpy as np

from earthkit.hydro.data_structures import RiverNetwork


def propagate(
    river_network: RiverNetwork,
    groups: np.ndarray,
    field: np.ndarray,
    invert_graph: bool,
    operation,
    *args,
    **kwargs,
):
    if invert_graph:
        for uid, did, eid in groups[::-1]:
            field = operation(field, did, uid, eid, *args, **kwargs)
    else:
        for did, uid, eid in groups:
            field = operation(field, did, uid, eid, *args, **kwargs)

    return field
