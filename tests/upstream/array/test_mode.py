import numpy as np
import pytest
from _test_inputs.readers import *

import earthkit.hydro as ekh


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
