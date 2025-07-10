import numpy as np

from earthkit.hydro.data_structures import RiverNetwork


def propagate(
    river_network: RiverNetwork,
    field: np.ndarray,
    invert_graph: bool,
    operation,
    *args,
    **kwargs,
):
    if invert_graph:
        for uid, did, eid in river_network.groups[::-1]:
            operation(field, did, uid, eid, *args, **kwargs)
    else:
        for did, uid, eid in river_network.groups:
            operation(field, did, uid, eid, *args, **kwargs)

    return field
