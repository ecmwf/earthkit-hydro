import numpy as np
import pytest
from _test_inputs.readers import *

import earthkit.hydro as ekh
from earthkit.hydro._readers import from_d8
from earthkit.hydro.data_structures import RiverNetwork

try:
    from earthkit.hydro import _rust  # noQA: F401

    RUST = True
except Exception:
    RUST = False


@pytest.mark.skipif(not RUST, reason="Rust unavailable")
def test_downstream_mode_manual_verification():
    """Test downstream mode with manually constructed network and verified expected output.

    This test creates a simple network with known topology and manually
    computes the expected mode values to robustly verify the implementation.

    Network:

    |-|-|-|x|

    Field:

    [1, 2, 1, 3]

    Expected downstream mode calculation (from sink backwards):
    - 3: mode([3]) = 3 (sink, only its own value)
    - 2: mode([1, 3]) = 1 (tie broken by smallest)
    - 1: mode([2, 1, 3]) = 1 (tie broken by smallest)
    - 0: mode([1, 2, 1, 3]) = 1 (appears twice)
    """
    d8 = np.array([[3, 3, 3, 5]])

    # Create river network
    river_network_storage = from_d8(d8)
    rn = RiverNetwork(river_network_storage)

    # Field values at each node
    field = np.array([1, 2, 1, 3], dtype=np.int64)

    # Expected mode values (manually computed for downstream)
    expected_mode = np.array([1, 1, 1, 3], dtype=np.int64)

    # Compute mode
    result = ekh.downstream.array.mode(rn, field, return_type="masked")

    # Verify exact match
    np.testing.assert_array_equal(
        result,
        expected_mode,
        err_msg=f"Mode mismatch: expected {expected_mode}, got {result}",
    )


@pytest.mark.skipif(not RUST, reason="Rust unavailable")
def test_downstream_mode_branching_network():
    r"""Test downstream mode with a branching network topology (V-shape confluence).

    Network:

    |||/|
    |-|x|

    Field:

    [3, 2]
    [3, 2]

    Expected downstream mode calculation (from sink backwards):
    Network structure: 0->2->3, 1->2->3
    - 3: mode([2]) = 2 (sink, only self)
    - 2: mode([3, 2]) = 2 (self + downstream, tie broken by smallest)
    - 1: mode([2, 3, 2]) = 2 (self + downstream, 2 appears twice)
    - 0: mode([3, 3, 2]) = 3 (self + downstream, 3 appears twice)
    """
    d8 = np.array([[8, 7], [6, 5]])

    river_network_storage = from_d8(d8)
    rn = RiverNetwork(river_network_storage)

    # Test field with known values
    field = np.array([3, 2, 3, 2], dtype=np.int64)

    expected_mode = np.array([3, 2, 2, 2], dtype=np.int64)

    result = ekh.downstream.array.mode(rn, field, return_type="masked")

    np.testing.assert_array_equal(
        result,
        expected_mode,
        err_msg=f"Branching network mode mismatch: expected {expected_mode}, got {result}",
    )


@pytest.mark.skipif(not RUST, reason="Rust unavailable")
def test_downstream_mode_complex_branching():
    r"""Test downstream mode with a more complex branching network.

    Network:

    |||||/|
    |-|-|x|

    Field:

    [1,1,2]
    [5,5,3]

    Expected downstream mode calculation (from sink backwards):
    Network structure: 0->3->4->5, 1->4->5, 2->4->5
    - 5: mode([3]) = 3 (sink, only self)
    - 4: mode([5, 3]) = 3 (tie broken by smallest)
    - 3: mode([5, 5, 3]) = 5 (5 appears twice)
    - 2: mode([2, 5, 3]) = 2 (tie broken by smallest)
    - 1: mode([1, 5, 3]) = 1 (tie broken by smallest)
    - 0: mode([1, 5, 5, 3]) = 5 (5 appears twice)
    """
    d8 = np.array([[8, 8, 7], [6, 6, 5]])

    river_network_storage = from_d8(d8)
    rn = RiverNetwork(river_network_storage)

    field = np.array([1, 1, 2, 5, 5, 3], dtype=np.int64)

    expected_mode = np.array([5, 1, 2, 5, 3, 3], dtype=np.int64)

    result = ekh.downstream.array.mode(rn, field, return_type="masked")

    np.testing.assert_array_equal(
        result,
        expected_mode,
        err_msg=f"Complex branching mode mismatch: expected {expected_mode}, got {result}",
    )


@pytest.mark.skipif(not RUST, reason="Rust unavailable")
def test_downstream_mode_complex_branching_negative():
    r"""Test downstream mode with a more complex branching network using negative values.

    Network:

    |||||/|
    |-|-|x|

    Field:

    [-1,-1,-2]
    [-5,-5,-3]

    Expected downstream mode calculation (from sink backwards):
    Network structure: 0->3->4->5, 1->4->5, 2->4->5
    - 5: mode([-3]) = -3 (sink, only self)
    - 4: mode([-5, -3]) = -5 (tie broken by smallest)
    - 3: mode([-5, -5, -3]) = -5 (appears twice)
    - 2: mode([-2, -5, -3]) = -5 (tie broken by smallest)
    - 1: mode([-1, -5, -3]) = -5 (tie broken by smallest)
    - 0: mode([-1, -5, -5, -3]) = -5 (appears twice)
    """
    d8 = np.array([[8, 8, 7], [6, 6, 5]])

    river_network_storage = from_d8(d8)
    rn = RiverNetwork(river_network_storage)

    field = -np.array([1, 1, 2, 5, 5, 3], dtype=np.int64)

    expected_mode = np.array([-5, -5, -5, -5, -5, -3], dtype=np.int64)

    result = ekh.downstream.array.mode(rn, field, return_type="masked")

    np.testing.assert_array_equal(
        result,
        expected_mode,
        err_msg=f"Complex branching mode mismatch: expected {expected_mode}, got {result}",
    )


@pytest.mark.skipif(not RUST, reason="Rust unavailable")
def test_downstream_mode_dominant_category():
    r"""Test downstream mode where one category clearly dominates.

    Network:

    |-|-|-|x|

    Field:

    [2, 2, 2, 1]

    Expected downstream mode calculation (from sink backwards):
    - 3: mode([1]) = 1 (sink, only self)
    - 2: mode([2, 1]) = 1 (tie broken by smallest)
    - 1: mode([2, 2, 1]) = 2 (2 appears twice)
    - 0: mode([2, 2, 2, 1]) = 2 (2 appears 3 times)
    """
    d8 = np.array([[3, 3, 3, 5]])

    river_network_storage = from_d8(d8)
    rn = RiverNetwork(river_network_storage)

    field = np.array([2, 2, 2, 1], dtype=np.int64)

    expected_mode = np.array([2, 2, 1, 1], dtype=np.int64)

    result = ekh.downstream.array.mode(rn, field, return_type="masked")

    np.testing.assert_array_equal(
        result,
        expected_mode,
        err_msg=f"Dominant category mismatch: expected {expected_mode}, got {result}",
    )


@pytest.mark.skipif(not RUST, reason="Rust unavailable")
@pytest.mark.parametrize(
    "river_network",
    [("cama_nextxy", cama_nextxy_1)],
    indirect=["river_network"],
)
def test_downstream_mode_constant(river_network):
    """Test downstream mode with a constant field - mode should be the constant value everywhere."""
    n_nodes = river_network.n_nodes

    # All nodes have the same value
    constant_value = 7
    input_field = np.full(n_nodes, constant_value, dtype=np.int64)

    # Compute mode
    output_field = ekh.downstream.array.mode(river_network, input_field, node_weights=None, return_type="masked")

    # All output values should be the constant value
    assert np.all(output_field == constant_value), f"Expected all {constant_value}, got {output_field}"


@pytest.mark.skipif(not RUST, reason="Rust unavailable")
@pytest.mark.parametrize(
    "river_network",
    [("cama_nextxy", cama_nextxy_1)],
    indirect=["river_network"],
)
def test_downstream_negative_non_consecutive_categories(river_network):
    """Test downstream mode with negative and non-consecutive category values."""
    input_field = np.array(
        [
            [-9, -9, -5, -23, -5],
            [-4, -23, -23, -4, -4],
            [-4, -23, -23, -4, -4],
            [-4, -23, -4, -4, -4],
        ],
        dtype=np.int64,
    )

    # Compute mode
    result = ekh.downstream.array.mode(river_network, input_field, node_weights=None, return_type="masked")

    # For downstream aggregation, we aggregate from sinks backwards
    # This requires manually computing the expected values based on the network topology
    # The expected values will be different from upstream
    expected_mode = np.array(
        [
            -4,
            -23,
            -23,
            -23,
            -23,
            -4,
            -23,
            -23,
            -23,
            -23,
            -23,
            -23,
            -23,
            -23,
            -23,
            -23,
            -23,
            -23,
            -4,
            -4,
        ],
        dtype=np.int64,
    )

    np.testing.assert_array_equal(
        result,
        expected_mode,
        err_msg=f"Dominant category mismatch: expected {expected_mode}, got {result}",
    )
