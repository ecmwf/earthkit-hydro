import numpy as np
import pytest
from _test_inputs.readers import *

import earthkit.hydro as ekh


@pytest.mark.parametrize(
    "river_network",
    [
        ("cama_nextxy", cama_nextxy_1),
    ],
    indirect=["river_network"],
)
def test_from_mask_node_mask(river_network):
    """Test creating a subnetwork with a node mask."""
    # Create a node mask that selects a subset of nodes
    node_mask = np.zeros(river_network.n_nodes, dtype=bool)
    node_mask[:10] = True  # Select first 10 nodes

    subnetwork = ekh.subnetwork.from_mask(river_network, node_mask=node_mask)

    # Check that the subnetwork has the correct number of nodes
    assert subnetwork.n_nodes == 10
    assert subnetwork.n_nodes < river_network.n_nodes

    # Check that the subnetwork is a different object
    assert subnetwork is not river_network


@pytest.mark.parametrize(
    "river_network",
    [
        ("cama_nextxy", cama_nextxy_1),
    ],
    indirect=["river_network"],
)
def test_from_mask_both_masks(river_network):
    """Test creating a subnetwork with both node and edge masks."""
    # Create masks for both nodes and edges
    node_mask = np.zeros(river_network.n_nodes, dtype=bool)
    node_mask[:10] = True  # Select first 10 nodes

    edge_mask = np.zeros(river_network.n_edges, dtype=bool)
    edge_mask[:5] = True  # Select first 5 edges

    subnetwork = ekh.subnetwork.from_mask(
        river_network, node_mask=node_mask, edge_mask=edge_mask
    )

    # Check that the subnetwork has fewer nodes and edges
    assert subnetwork.n_nodes <= 10
    assert subnetwork.n_edges <= 5
    assert subnetwork.n_nodes < river_network.n_nodes


@pytest.mark.parametrize(
    "river_network",
    [
        ("cama_nextxy", cama_nextxy_1),
    ],
    indirect=["river_network"],
)
def test_from_mask_no_mask(river_network):
    """Test creating a subnetwork with no masks (should return a copy)."""
    subnetwork = ekh.subnetwork.from_mask(river_network)

    # Check that it's a copy with the same properties
    assert subnetwork.n_nodes == river_network.n_nodes
    assert subnetwork.n_edges == river_network.n_edges
    assert subnetwork is not river_network


@pytest.mark.parametrize(
    "river_network",
    [
        ("cama_nextxy", cama_nextxy_1),
    ],
    indirect=["river_network"],
)
def test_crop(river_network):
    """Test cropping a gridded network to minimum bounding box."""
    # Skip test if river network doesn't have coords (required for crop)
    if river_network._storage.coords is None:
        pytest.skip("River network does not have coordinates required for crop")

    # First create a subnetwork with a specific node mask that leaves empty borders
    # This ensures the crop will actually reduce the grid dimensions
    node_mask = np.zeros(river_network.n_nodes, dtype=bool)
    # Select nodes in the middle of the grid to leave empty rows/columns at edges
    # Use a smaller subset to ensure there's space to crop
    n_select = min(10, river_network.n_nodes // 2)
    node_mask[:n_select] = True

    subnetwork = ekh.subnetwork.from_mask(river_network, node_mask=node_mask)

    # Skip if subnetwork doesn't have coords
    if subnetwork._storage.coords is None:
        pytest.skip("Subnetwork does not have coordinates required for crop")

    # Now crop the subnetwork
    cropped = ekh.subnetwork.crop(subnetwork)

    # Check that number of nodes is preserved (crop doesn't remove nodes)
    assert cropped.n_nodes == subnetwork.n_nodes

    # Check that the grid dimensions are reduced (actual cropping happened)
    assert (cropped.shape[0] < subnetwork.shape[0]) or (cropped.shape[1] < subnetwork.shape[1])

    # Check that it's a different object
    assert cropped is not subnetwork
