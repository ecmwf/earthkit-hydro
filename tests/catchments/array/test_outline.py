# SPDX-FileCopyrightText: 2026- European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from _test_inputs.catchment import *
from _test_inputs.readers import *

import earthkit.hydro as ekh
from earthkit.hydro._utils.outline import cell_edges, outline_cells


def rings_of(outline):
    """Splits a NaN-separated outline into its individual closed rings."""
    breaks = np.nonzero(np.isnan(outline[:, 0]))[0]
    return [np.delete(ring, np.isnan(ring[:, 0]), axis=0) for ring in np.split(outline, breaks)]


def signed_area(ring):
    """Shoelace area of a closed ring, positive for one winding and negative for the other."""
    return 0.5 * np.sum(ring[:-1, 1] * ring[1:, 0] - ring[1:, 1] * ring[:-1, 0])


def outline_of_mask(mask):
    """Outlines a boolean mask on a grid whose cell centres are 0, 1, 2, ..."""
    mask = np.asarray(mask, dtype=bool)
    rows, cols = np.nonzero(mask)
    return outline_cells(
        rows,
        cols,
        mask.shape,
        cell_edges(np.arange(mask.shape[0])),
        cell_edges(np.arange(mask.shape[1])),
    )


@pytest.mark.parametrize(
    "mask, expected_areas",
    [
        ([[0, 0, 0], [0, 1, 0], [0, 0, 0]], [1]),  # single cell
        ([[1, 1], [1, 1]], [4]),  # whole grid, boundary runs along the grid edge
        ([[1, 0, 0], [1, 1, 0], [0, 1, 1]], [5]),  # staircase
        ([[1, 1, 1], [1, 0, 1], [1, 1, 1]], [9, -1]),  # hole in the middle
        ([[1, 0], [0, 1]], [1, 1]),  # cells touching only diagonally
    ],
)
def test_outline_cells_rings(mask, expected_areas):
    """Each ring is closed and encloses the expected area, largest ring first."""
    rings = rings_of(outline_of_mask(mask))

    assert len(rings) == len(expected_areas)
    for ring, expected_area in zip(rings, expected_areas):
        np.testing.assert_array_equal(ring[0], ring[-1])
        assert signed_area(ring) == expected_area

    # The rings together enclose exactly the cells of the mask, so a hole must wind
    # the opposite way to the ring around it.
    assert sum(signed_area(ring) for ring in rings) == np.sum(mask)


def test_outline_cells_keeps_only_turning_corners():
    outline = outline_of_mask(np.ones((5, 7)))

    np.testing.assert_array_equal(
        outline,
        [[-0.5, -0.5], [-0.5, 6.5], [4.5, 6.5], [4.5, -0.5], [-0.5, -0.5]],
    )


def test_cell_edges_handles_non_uniform_spacing():
    np.testing.assert_allclose(cell_edges([0.0, 1.0, 3.0, 4.0]), [-0.5, 0.5, 2.0, 3.5, 4.5])


def test_cell_edges_rejects_short_axis():
    with pytest.raises(ValueError):
        cell_edges([0.0])


@pytest.mark.parametrize(
    "river_network, locations",
    [
        (("cama_nextxy", cama_nextxy_1), catchment_query_field_1),
        (("cama_nextxy", cama_nextxy_2), catchment_query_field_2),
    ],
    indirect=["river_network"],
)
@pytest.mark.parametrize("overwrite", [True, False])
def test_outline_matches_find(river_network, locations, overwrite):
    """The outline of each catchment encloses exactly the cells `find` labels."""
    labels = ekh.catchments.array.find(river_network, locations, overwrite=overwrite, return_type="gridded")
    outlines = ekh.catchments.array.outline(river_network, locations, overwrite=overwrite)

    assert len(outlines) == len(locations)
    for label, outline in enumerate(outlines):
        n_cells = np.sum(labels == label)
        assert sum(signed_area(ring) for ring in rings_of(outline)) == n_cells
        if n_cells == 0:
            assert outline.shape == (0, 2)


@pytest.mark.parametrize(
    "river_network, locations",
    [(("cama_nextxy", cama_nextxy_1), catchment_query_field_1)],
    indirect=["river_network"],
)
def test_outline_of_list_locations(river_network, locations):
    outlines = ekh.catchments.array.outline(river_network, locations)

    assert isinstance(outlines, list)
    # Coordinate pairs are in grid axis order, on the edges between cell centres.
    for outline in outlines:
        assert outline.shape[1] == 2
        for ring in rings_of(outline):
            assert np.all(np.isin(ring[:, 0], cell_edges(river_network.coords["y"])))
            assert np.all(np.isin(ring[:, 1], cell_edges(river_network.coords["x"])))


@pytest.mark.parametrize(
    "river_network, locations",
    [(("cama_nextxy", cama_nextxy_1), catchment_query_field_1)],
    indirect=["river_network"],
)
def test_outline_of_dict_locations(river_network, locations):
    """A dict of named points gives a dict of outlines keyed the same way."""
    rows, cols = np.divmod(river_network.mask[locations], river_network.shape[1])
    named = {
        f"station_{node}": (river_network.coords["y"][row], river_network.coords["x"][col])
        for node, row, col in zip(locations, rows, cols)
    }

    outlines = ekh.catchments.array.outline(river_network, named)

    assert list(outlines.keys()) == list(named.keys())
    for name, outline in zip(named, ekh.catchments.array.outline(river_network, locations)):
        np.testing.assert_array_equal(outlines[name], outline)


@pytest.mark.parametrize(
    "river_network, locations",
    [(("cama_nextxy", cama_nextxy_1), catchment_query_field_1)],
    indirect=["river_network"],
)
def test_outline_toplevel_matches_array(river_network, locations):
    for expected, result in zip(
        ekh.catchments.array.outline(river_network, locations),
        ekh.catchments.outline(river_network, locations),
    ):
        np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "river_network, locations",
    [(("cama_nextxy", cama_nextxy_1), catchment_query_field_1)],
    indirect=["river_network"],
)
def test_outline_requires_coords(river_network, locations):
    river_network.coords = None
    with pytest.raises(ValueError):
        ekh.catchments.array.outline(river_network, locations)


@pytest.mark.parametrize(
    "river_network, locations",
    [(("cama_nextxy", cama_nextxy_1), catchment_query_field_1)],
    indirect=["river_network"],
)
def test_outline_rejects_vector_network(river_network, locations):
    river_network.shape = None
    with pytest.raises(NotImplementedError):
        ekh.catchments.array.outline(river_network, locations)


@pytest.mark.parametrize(
    "river_network, locations",
    [(("cama_nextxy", cama_nextxy_1), catchment_query_field_1)],
    indirect=["river_network"],
)
def test_outline_rejects_non_numpy_backend(river_network, locations):
    river_network.array_backend = "torch"
    with pytest.raises(NotImplementedError):
        ekh.catchments.array.outline(river_network, locations)
