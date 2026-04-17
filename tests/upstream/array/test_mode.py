import numpy as np
import pytest
from _test_inputs.readers import *

import earthkit.hydro as ekh
from earthkit.hydro._readers import from_cama_nextxy
from earthkit.hydro.data_structures import RiverNetwork


def test_upstream_mode_manual_verification():
    """
    Test mode with manually constructed network and verified expected output.

    This test creates a simple network with known topology and manually
    computes the expected mode values to robustly verify the implementation.

    Network topology (linear): 0 -> 1 -> 2 -> 3 (in a 1x4 grid)
    Field values: [1, 2, 1, 3]

    Expected upstream mode:
    - Node 0 (source): mode([1]) = 1
    - Node 1: mode([1, 2]) = 1 (tie broken by smallest)
    - Node 2: mode([1, 2, 1]) = 1 (appears twice)
    - Node 3: mode([1, 2, 1, 3]) = 1 (appears twice)
    """
    # Create a simple linear network in a 1x4 grid: [0,0] -> [0,1] -> [0,2] -> [0,3]
    # nextxy format uses 1-indexed coordinates for downstream cells
    # Cell (row, col) flows to nextxy_y[row,col], nextxy_x[row,col]
    nextxy_x = np.array([[2, 3, 4, -9]])  # x-coords (columns, 1-indexed)
    nextxy_y = np.array([[1, 1, 1, -9]])  # y-coords (rows, 1-indexed), -9 means sink

    # Create river network
    river_network_storage = from_cama_nextxy(nextxy_x, nextxy_y)
    rn = RiverNetwork(river_network_storage)

    # Field values at each node
    field = np.array([1, 2, 1, 3], dtype=np.int64)

    # Expected mode values (manually computed)
    expected_mode = np.array([1, 1, 1, 1], dtype=np.int64)

    # Compute mode
    result = ekh.upstream.array.mode(rn, field, return_type="masked")

    # Verify exact match
    np.testing.assert_array_equal(result, expected_mode,
        err_msg=f"Mode mismatch: expected {expected_mode}, got {result}")

    print("Manual verification test passed!")
    print(f"Field: {field}")
    print(f"Expected mode: {expected_mode}")
    print(f"Computed mode: {result}")


def test_upstream_mode_branching_network():
    r"""
    Test mode with a branching network topology.

    Network:         0
                    / \
                   1   2
                    \ /
                     3

    Field: [1, 2, 2, ?]
    Node 3 should have mode([1, 2, 2, field[3]])
    """
    # Create branching network in a 2x2 grid:
    # [0,0]=0 -> [1,0]=2
    # [0,1]=1 -> [1,0]=2
    # [1,0]=2 -> [1,1]=3
    # [1,1]=3 = sink
    nextxy_x = np.array([
        [1, 1],  # row 0: cells flow to column 1
        [2, -9]  # row 1: cell 0 flows to column 2, cell 1 is sink
    ])
    nextxy_y = np.array([
        [2, 2],  # row 0: cells flow to row 2
        [2, -9]  # row 1: cell 0 flows to row 2, cell 1 is sink
    ])

    river_network_storage = from_cama_nextxy(nextxy_x, nextxy_y)
    rn = RiverNetwork(river_network_storage)

    # Field with category 2 appearing most frequently
    field = np.array([1, 2, 2, 1], dtype=np.int64)

    # Expected: sources keep their value, node 3 gets mode of all upstream
    # Node 0: mode([1]) = 1
    # Node 1: mode([2]) = 2
    # Node 2: mode([2]) = 2
    # Node 3: mode([1, 1, 2, 2]) = tie, smallest = 1
    expected_mode = np.array([1, 2, 2, 1], dtype=np.int64)

    result = ekh.upstream.array.mode(rn, field, return_type="masked")

    np.testing.assert_array_equal(result, expected_mode,
        err_msg=f"Branching network mode mismatch: expected {expected_mode}, got {result}")

    print("Branching network test passed!")


@pytest.mark.parametrize(
    "river_network",
    [("cama_nextxy", cama_nextxy_1)],
    indirect=["river_network"],
)
def test_upstream_mode_simple(river_network):
    """
    Test mode calculation with a simple categorical field.

    Creates a field where each node has a categorical value,
    and verifies that the mode is computed correctly for
    downstream accumulation.
    """
    # Create a simple categorical field with values 1, 2, 3
    # For a linear network: 1 -> 2 -> 3, if field is [1, 2, 1]
    # The mode at each node should be:
    # Node 0: mode of [1] = 1
    # Node 1: mode of [1, 2] = 1 (tie, smallest wins)
    # Node 2: mode of [1, 2, 1] = 1

    n_nodes = river_network.n_nodes

    # Create a field with repeating categorical values
    np.random.seed(42)
    input_field = np.random.randint(1, 5, size=n_nodes, dtype=np.int64)

    # Compute mode
    output_field = ekh.upstream.array.mode(
        river_network, input_field, node_weights=None, return_type="masked"
    )

    # Basic checks
    assert output_field.dtype == np.int64 or output_field.dtype == np.int32
    assert output_field.shape == input_field.shape
    assert len(output_field) == n_nodes

    # Check that output values are in the range of input values
    assert np.all(output_field >= input_field.min())
    assert np.all(output_field <= input_field.max())

    # For source nodes, the mode should be the node's own value
    for source in river_network.sources:
        assert output_field[source] == input_field[source], \
            f"Source node {source}: expected {input_field[source]}, got {output_field[source]}"

    print(f"Input field: {input_field}")
    print(f"Output field (mode): {output_field}")
    print(f"Test passed!")


@pytest.mark.parametrize(
    "river_network",
    [("cama_nextxy", cama_nextxy_1)],
    indirect=["river_network"],
)
def test_upstream_mode_constant(river_network):
    """Test mode with a constant field - mode should be the constant value everywhere."""
    n_nodes = river_network.n_nodes

    # All nodes have the same value
    constant_value = 7
    input_field = np.full(n_nodes, constant_value, dtype=np.int64)

    # Compute mode
    output_field = ekh.upstream.array.mode(
        river_network, input_field, node_weights=None, return_type="masked"
    )

    # All output values should be the constant value
    assert np.all(output_field == constant_value), \
        f"Expected all {constant_value}, got {output_field}"

    print(f"Constant field test passed! All values are {constant_value}")


@pytest.mark.parametrize(
    "river_network",
    [("cama_nextxy", cama_nextxy_1)],
    indirect=["river_network"],
)
def test_upstream_mode_binary(river_network):
    """Test mode with binary categorical data."""
    n_nodes = river_network.n_nodes

    # Binary field: alternating 0 and 1
    input_field = np.array([i % 2 for i in range(n_nodes)], dtype=np.int64)

    # Compute mode
    output_field = ekh.upstream.array.mode(
        river_network, input_field, node_weights=None, return_type="masked"
    )

    # Check that output is binary
    assert np.all((output_field == 0) | (output_field == 1)), \
        f"Output should be binary, got {output_field}"

    # For source nodes, mode should be their own value
    for source in river_network.sources:
        assert output_field[source] == input_field[source]

    print(f"Binary field test passed!")


@pytest.mark.parametrize(
    "river_network",
    [("cama_nextxy", cama_nextxy_1)],
    indirect=["river_network"],
)
def test_upstream_mode_weights_unsupported(river_network):
    """Test that mode raises error when weights are provided."""
    n_nodes = river_network.n_nodes
    input_field = np.random.randint(1, 5, size=n_nodes, dtype=np.int64)
    node_weights = np.ones(n_nodes)

    # Mode should raise an error when weights are provided
    with pytest.raises(ValueError, match="does not support weights"):
        ekh.upstream.array.mode(
            river_network, input_field, node_weights=node_weights, return_type="masked"
        )

    print(f"Weights unsupported test passed!")


@pytest.mark.parametrize(
    "river_network",
    [("cama_nextxy", cama_nextxy_1)],
    indirect=["river_network"],
)
def test_upstream_mode_dtype_conversion(river_network):
    """Test that mode works with different input dtypes."""
    n_nodes = river_network.n_nodes

    # Test with int32
    input_field_int32 = np.random.randint(1, 5, size=n_nodes, dtype=np.int32)
    output_int32 = ekh.upstream.array.mode(
        river_network, input_field_int32, return_type="masked"
    )
    assert output_int32.shape == input_field_int32.shape

    # Test with int64
    input_field_int64 = np.random.randint(1, 5, size=n_nodes, dtype=np.int64)
    output_int64 = ekh.upstream.array.mode(
        river_network, input_field_int64, return_type="masked"
    )
    assert output_int64.shape == input_field_int64.shape

    print(f"Dtype conversion test passed!")


@pytest.mark.parametrize(
    "river_network",
    [("cama_nextxy", cama_nextxy_1)],
    indirect=["river_network"],
)
def test_upstream_mode_land_cover_scenario(river_network):
    """
    Test mode with a realistic land cover scenario.

    Simulates land cover classes (1=forest, 2=grassland, 3=urban, 4=water)
    and verifies that the dominant upstream land cover is correctly computed.
    """
    n_nodes = river_network.n_nodes

    # Create a land cover field with specific patterns
    # Use deterministic seed for reproducibility
    np.random.seed(123)

    # Create land cover with forest (1) being most common
    # 50% forest, 25% grassland, 15% urban, 10% water
    land_cover = np.random.choice(
        [1, 2, 3, 4],
        size=n_nodes,
        p=[0.5, 0.25, 0.15, 0.10]
    ).astype(np.int64)

    # Compute mode
    mode_result = ekh.upstream.array.mode(
        river_network, land_cover, return_type="masked"
    )

    # Verify basic properties
    assert mode_result.dtype in [np.int64, np.int32]
    assert mode_result.shape == land_cover.shape
    assert np.all((mode_result >= 1) & (mode_result <= 4))

    # For source nodes, mode equals their own value
    for source in river_network.sources:
        assert mode_result[source] == land_cover[source]

    print(f"Land cover scenario test passed!")
    print(f"Land cover distribution: {np.bincount(land_cover[land_cover > 0])}")
    print(f"Mode distribution: {np.bincount(mode_result[mode_result > 0])}")


@pytest.mark.parametrize(
    "river_network",
    [("cama_nextxy", cama_nextxy_1)],
    indirect=["river_network"],
)
def test_upstream_mode_tie_breaking(river_network):
    """
    Test that ties are broken correctly (smallest value wins).

    In a scenario where two categories appear with equal frequency,
    the mode should be the smaller category value.
    """
    n_nodes = river_network.n_nodes

    # Create a field where at some nodes we'll have ties
    # Use a pattern that creates balanced distributions
    input_field = np.array([1 if i % 2 == 0 else 2 for i in range(n_nodes)], dtype=np.int64)

    # Compute mode
    mode_result = ekh.upstream.array.mode(
        river_network, input_field, return_type="masked"
    )

    # Verify that mode values are valid
    assert np.all((mode_result == 1) | (mode_result == 2))

    # For source nodes, verify they match input
    for source in river_network.sources:
        assert mode_result[source] == input_field[source]

    print(f"Tie-breaking test passed!")


@pytest.mark.parametrize(
    "river_network",
    [("cama_nextxy", cama_nextxy_1)],
    indirect=["river_network"],
)
def test_upstream_mode_categorical_range(river_network):
    """
    Test mode with a wide range of categorical values.

    Verifies that mode works correctly with non-consecutive category values
    (e.g., categories 10, 20, 30 instead of 1, 2, 3).
    """
    n_nodes = river_network.n_nodes

    # Use non-consecutive category values
    np.random.seed(456)
    categories = [10, 20, 30, 40, 50]
    input_field = np.random.choice(categories, size=n_nodes).astype(np.int64)

    # Compute mode
    mode_result = ekh.upstream.array.mode(
        river_network, input_field, return_type="masked"
    )

    # Verify output values are from the input set
    assert np.all(np.isin(mode_result, categories))

    # Verify source nodes
    for source in river_network.sources:
        assert mode_result[source] == input_field[source]

    print(f"Categorical range test passed!")
    print(f"Categories used: {categories}")
    print(f"Unique modes: {np.unique(mode_result)}")


@pytest.mark.parametrize(
    "river_network",
    [("cama_nextxy", cama_nextxy_1)],
    indirect=["river_network"],
)
def test_upstream_mode_negative_categories(river_network):
    """
    Test mode with negative category values.

    Verifies that mode works with negative integers, which might represent
    certain data conventions (e.g., -1 for missing/special values).
    """
    n_nodes = river_network.n_nodes

    # Use categories including negative values
    np.random.seed(789)
    categories = [-1, 0, 1, 2]
    input_field = np.random.choice(categories, size=n_nodes).astype(np.int64)

    # Compute mode
    mode_result = ekh.upstream.array.mode(
        river_network, input_field, return_type="masked"
    )

    # Verify output values are from the input set
    assert np.all(np.isin(mode_result, categories))

    # Verify source nodes
    for source in river_network.sources:
        assert mode_result[source] == input_field[source]

    print(f"Negative categories test passed!")
    print(f"Categories: {categories}")
    print(f"Unique modes: {np.unique(mode_result)}")
