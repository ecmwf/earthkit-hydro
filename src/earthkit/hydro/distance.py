import numpy as np

from .core import flow
from .utils import is_missing, mask_and_unmask_data


@mask_and_unmask_data
def compute_distance(
    river_network,
    field,
    mv=-1,
    in_place=False,
    allow_downstream=True,
    allow_upstream=True,
):
    # Note: due to no bifurcation assumption
    # distance can be calculated by
    # 1. going downstream
    # 2. then going upstream

    if not in_place:
        field = field.copy()

    if len(field.shape) == 1:
        if allow_downstream:
            flow(river_network, field, False, _downstream_distance_2D, mv)
        if allow_upstream:
            flow(river_network, field, True, _upstream_distance_2D, mv)
    else:
        if allow_downstream:
            flow(river_network, field, False, _downstream_distance_ND, mv)
        if allow_upstream:
            flow(river_network, field, True, _upstream_distance_ND, mv)

    return field


def _upstream_distance_2D(river_network, field, grouping, mv):
    missing_upstream = is_missing(field[grouping], mv)
    missing_current = is_missing(field[river_network.downstream_nodes[grouping]], mv)
    replace_upstream = grouping[missing_upstream & ~missing_current]
    check_shorter = grouping[~missing_upstream & ~missing_current]

    if replace_upstream.size != 0:
        # replace upstream with downstream + 1 (if upstream missing)
        field[replace_upstream] = (
            np.max(field[river_network.downstream_nodes[replace_upstream]]) + 1
        )
        np.minimum.at(
            field,
            replace_upstream,
            field[river_network.downstream_nodes[replace_upstream]] + 1,
        )
    if check_shorter.size != 0:
        # replace upstream with min (upstream, downstream + 1)
        np.minimum.at(
            field,
            check_shorter,
            field[river_network.downstream_nodes[check_shorter]] + 1,
        )


def _upstream_distance_ND(river_network, field, grouping, mv):
    missing_upstream = is_missing(field[grouping], mv)
    missing_current = is_missing(field[river_network.downstream_nodes[grouping]], mv)
    replace_upstream_mask = missing_upstream & ~missing_current
    check_shorter_mask = ~missing_upstream & ~missing_current
    replace_upstream_indices = np.array(np.where(replace_upstream_mask))
    check_shorter_indices = np.array(np.where(check_shorter_mask))
    replace_upstream_indices[0] = grouping[replace_upstream_indices[0]]
    check_shorter_indices[0] = grouping[check_shorter_indices[0]]
    downstream_replace_upstream = replace_upstream_indices.copy()
    downstream_replace_upstream[0] = river_network.downstream_nodes[
        downstream_replace_upstream[0]
    ]
    downstream_check_shorter = check_shorter_indices.copy()
    downstream_check_shorter[0] = river_network.downstream_nodes[
        downstream_check_shorter[0]
    ]
    field[replace_upstream_indices] = np.max(field[downstream_replace_upstream]) + 1
    np.minimum.at(
        field, replace_upstream_indices, field[downstream_replace_upstream] + 1
    )
    np.minimum.at(field, check_shorter_indices, field[downstream_check_shorter] + 1)


def _downstream_distance_2D(river_network, field, grouping, mv):
    missing_downstream = is_missing(field[river_network.downstream_nodes[grouping]], mv)
    missing_current = is_missing(field[grouping], mv)
    replace_downstream = grouping[missing_downstream & ~missing_current]
    check_shorter = grouping[~missing_downstream & ~missing_current]
    if replace_downstream.size != 0:
        # replace downstream with upstream + 1 (if downstream missing)
        field[river_network.downstream_nodes[replace_downstream]] = (
            np.max(field[replace_downstream]) + 1
        )
        np.minimum.at(
            field,
            river_network.downstream_nodes[replace_downstream],
            field[replace_downstream] + 1,
        )
    if check_shorter.size != 0:
        # replace downstream with min (downstream, upstream + 1)
        np.minimum.at(
            field,
            river_network.downstream_nodes[check_shorter],
            field[check_shorter] + 1,
        )


def _downstream_distance_ND(river_network, field, grouping, mv):
    missing_downstream = is_missing(field[river_network.downstream_nodes[grouping]], mv)
    missing_current = is_missing(field[grouping], mv)
    replace_downstream_mask = missing_downstream & ~missing_current
    check_shorter_mask = ~missing_downstream & ~missing_current
    replace_downstream_indices = np.array(np.where(replace_downstream_mask))
    check_shorter_indices = np.array(np.where(check_shorter_mask))
    replace_downstream_indices[0] = grouping[replace_downstream_indices[0]]
    check_shorter_indices[0] = grouping[check_shorter_indices[0]]
    downstream_replace_downstream = replace_downstream_indices.copy()
    downstream_replace_downstream[0] = river_network.downstream_nodes[
        downstream_replace_downstream[0]
    ]
    downstream_check_shorter = check_shorter_indices.copy()
    downstream_check_shorter[0] = river_network.downstream_nodes[
        downstream_check_shorter[0]
    ]
    field[downstream_replace_downstream] = np.max(field[replace_downstream_indices]) + 1
    np.minimum.at(
        field, downstream_replace_downstream, field[replace_downstream_indices] + 1
    )
    np.minimum.at(field, downstream_check_shorter, field[check_shorter_indices] + 1)
