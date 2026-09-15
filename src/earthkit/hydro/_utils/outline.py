# SPDX-FileCopyrightText: 2026- European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

from itertools import pairwise

import numpy as np

# Each side of a cell (i, j), as (row offset, col offset) of the neighbouring cell
# followed by the (row, col) corner offsets of the directed edge along that side.
# Corner (i, j) is the corner shared by the top-left of cell (i, j), so the edges
# below wind clockwise in index space (top -> right -> bottom -> left).
_SIDES = (
    ((-1, 0), (0, 0), (0, 1)),  # top
    ((0, 1), (0, 1), (1, 1)),  # right
    ((1, 0), (1, 1), (1, 0)),  # bottom
    ((0, -1), (1, 0), (0, 0)),  # left
)


def cell_edges(centres):
    """
    Converts cell centre coordinates to cell edge coordinates.

    Parameters
    ----------
    centres : array-like
        Coordinates of `n` cell centres along one grid axis.

    Returns
    -------
    numpy.ndarray
        The `n + 1` coordinates of the cell edges. Interior edges are the midpoints
        between neighbouring centres, so non-uniform spacing is handled exactly.
    """
    centres = np.asarray(centres, dtype=float)
    if centres.size < 2:
        raise ValueError("Cannot compute cell edges for a grid axis of length < 2.")
    interior = (centres[:-1] + centres[1:]) / 2
    return np.concatenate(
        [
            [2 * centres[0] - interior[0]],
            interior,
            [2 * centres[-1] - interior[-1]],
        ]
    )


def _boundary_edges(rows, cols, shape):
    """
    Finds the directed edges tracing the outside of a set of gridcells.

    Parameters
    ----------
    rows, cols : numpy.ndarray
        Row and column indices of the cells in the region.
    shape : tuple
        The shape of the grid the cells live on.

    Returns
    -------
    tuple of numpy.ndarray
        Two `(n_edges, 2)` arrays holding the start and end corner indices of each
        directed unit edge. The edges wind clockwise in index space.
    """
    n_rows, n_cols = shape
    cell_codes = rows * n_cols + cols

    starts, ends = [], []
    for (d_row, d_col), start_offset, end_offset in _SIDES:
        neighbour_rows, neighbour_cols = rows + d_row, cols + d_col
        in_grid = (neighbour_rows >= 0) & (neighbour_rows < n_rows) & (neighbour_cols >= 0) & (neighbour_cols < n_cols)
        neighbour_codes = np.where(in_grid, neighbour_rows * n_cols + neighbour_cols, -1)
        exposed = ~np.isin(neighbour_codes, cell_codes)
        exposed_rows, exposed_cols = rows[exposed], cols[exposed]
        starts.append(np.stack([exposed_rows + start_offset[0], exposed_cols + start_offset[1]], axis=-1))
        ends.append(np.stack([exposed_rows + end_offset[0], exposed_cols + end_offset[1]], axis=-1))

    return np.concatenate(starts), np.concatenate(ends)


def _trace_rings(starts, ends, n_cols):
    """
    Chains directed boundary edges into closed rings.

    Parameters
    ----------
    starts, ends : numpy.ndarray
        `(n_edges, 2)` arrays of start and end corner indices, as returned by
        `_boundary_edges`.
    n_cols : int
        The number of columns of the grid, used to encode corners as integers.

    Returns
    -------
    list of numpy.ndarray
        The edge indices making up each ring, in traversal order.
    """
    stride = n_cols + 1
    start_codes = starts[:, 0] * stride + starts[:, 1]
    end_codes = ends[:, 0] * stride + ends[:, 1]

    order = np.argsort(start_codes, kind="stable")
    sorted_start_codes = start_codes[order]
    first = np.searchsorted(sorted_start_codes, end_codes, side="left")
    last = np.searchsorted(sorted_start_codes, end_codes, side="right")

    # Every edge ends where at least one other edge starts, so the first candidate
    # is always a valid successor.
    successors = order[first]

    directions = ends - starts
    for edge in np.nonzero(last - first > 1)[0]:
        # A corner where two cells touch only diagonally starts two edges. Take the
        # sharpest clockwise turn, so each of the two cells keeps its own ring.
        wanted = (directions[edge, 1], -directions[edge, 0])
        for candidate in order[first[edge] : last[edge]]:
            if tuple(directions[candidate]) == wanted:
                successors[edge] = candidate
                break

    rings = []
    visited = np.zeros(starts.shape[0], dtype=bool)
    for edge in range(starts.shape[0]):
        if visited[edge]:
            continue
        ring = []
        while not visited[edge]:
            visited[edge] = True
            ring.append(edge)
            edge = successors[edge]
        rings.append(np.asarray(ring))

    return rings


def _drop_collinear(ring):
    """
    Drops the points of a closed ring that lie in the middle of a straight run.

    Parameters
    ----------
    ring : numpy.ndarray
        An `(n_edges + 1, 2)` array of corner indices whose first point is repeated
        as its last.

    Returns
    -------
    numpy.ndarray
        The same ring, keeping only the corners at which it changes direction.
    """
    steps = np.diff(ring, axis=0)
    # np.roll lines each step up with the one before it, wrapping around the ring.
    corners = ring[:-1][np.any(steps != np.roll(steps, 1, axis=0), axis=-1)]
    return np.append(corners, corners[:1], axis=0)


def _enclosed_area(ring):
    """
    Computes the area a closed ring encloses, in gridcells.

    Parameters
    ----------
    ring : numpy.ndarray
        An `(n_corners + 1, 2)` array of corner indices whose first point is
        repeated as its last.

    Returns
    -------
    int
        The enclosed area. Always positive, so rings tracing a hole (which wind the
        other way) compare against outer rings on equal terms.
    """
    shoelace = ring[:-1, 1] * ring[1:, 0] - ring[1:, 1] * ring[:-1, 0]
    return abs(int(shoelace.sum())) // 2


def outline_cells(rows, cols, shape, axis0_edges, axis1_edges):
    """
    Traces the outline of a set of gridcells.

    Parameters
    ----------
    rows, cols : numpy.ndarray
        Row and column indices of the cells in the region.
    shape : tuple
        The shape of the grid the cells live on.
    axis0_edges, axis1_edges : numpy.ndarray
        Cell edge coordinates along each grid axis, as returned by `cell_edges`.

    Returns
    -------
    numpy.ndarray
        An `(n_points, 2)` array of coordinate pairs tracing the region boundary.
        Only the corners at which the boundary turns are kept. Each ring is closed
        (its first point is repeated as its last) and rings are ordered by
        decreasing enclosed area and separated by a row of NaNs.
    """
    if rows.size == 0:
        return np.empty((0, 2), dtype=float)

    starts, ends = _boundary_edges(rows, cols, shape)
    rings = _trace_rings(starts, ends, shape[1])

    closed_rings = []
    for ring in rings:
        corners = _drop_collinear(starts[np.append(ring, ring[0])])
        closed_rings.append((_enclosed_area(corners), corners))
    # Largest first, so the outer ring of a region with holes comes out in front.
    closed_rings.sort(key=lambda ring: -ring[0])

    separator = np.full((1, 2), np.nan)
    points = [
        part
        for _, corners in closed_rings
        for part in (separator, np.stack([axis0_edges[corners[:, 0]], axis1_edges[corners[:, 1]]], axis=-1))
    ]
    return np.concatenate(points[1:])


def outline_labels(labels, n_labels, flat_indices, shape, coords):
    """
    Traces the outline of every label of a labelled set of gridcells.

    Parameters
    ----------
    labels : numpy.ndarray
        Integer label in `0, ..., n_labels - 1` for each cell.
    n_labels : int
        The number of labels. Labels with no cells give an empty outline.
    flat_indices : numpy.ndarray
        Flattened index on the grid of each cell.
    shape : tuple
        The shape of the grid the cells live on.
    coords : sequence of array-like
        Cell centre coordinates along each grid axis, in grid axis order.

    Returns
    -------
    list of numpy.ndarray
        One `(n_points, 2)` array of coordinate pairs per label, as returned by
        `outline_cells`.
    """
    rows, cols = np.divmod(flat_indices, shape[1])
    axis0_edges, axis1_edges = (cell_edges(values) for values in coords)

    order = np.argsort(labels, kind="stable")
    splits = np.searchsorted(labels[order], np.arange(n_labels + 1))

    return [
        outline_cells(rows[cells], cols[cells], shape, axis0_edges, axis1_edges)
        for cells in (order[start:stop] for start, stop in pairwise(splits))
    ]
