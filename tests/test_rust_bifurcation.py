import time
from types import SimpleNamespace

import numpy as np
import pytest

import earthkit.hydro as ekh

try:
    from earthkit.hydro import _rust  # noQA: F401

    RUST = True
except ImportError:
    RUST = False


def network(groups, bifurcates):
    return SimpleNamespace(
        groups=groups,
        bifurcates=bifurcates,
        array_backend="numpy",
        return_type="masked",
        shape=None,
    )


@pytest.mark.skipif(not RUST, reason="Rust unavailable")
def test_percentile_supports_bifurcation_in_both_directions():
    groups = [
        np.array([[1], [0]], dtype=np.int64),
        np.array([[2, 3], [1, 1]], dtype=np.int64),
    ]
    river_network = network(groups, bifurcates=True)
    field = np.array([0.0, 10.0, 20.0, 30.0])

    upstream = ekh.upstream.array.percentile(river_network, field, p=0.5, return_type="masked")
    downstream = ekh.downstream.array.percentile(river_network, field, p=0.5, return_type="masked")

    np.testing.assert_allclose(upstream, [0.0, 5.0, 10.0, 10.0])
    np.testing.assert_allclose(downstream, [15.0, 20.0, 20.0, 30.0])


@pytest.mark.skipif(not RUST, reason="Rust unavailable")
def test_downstream_retains_source_used_across_groups():
    groups = [
        np.array([[1, 3], [0, 2]], dtype=np.int64),
        np.array([[3], [1]], dtype=np.int64),
        np.array([[4], [3]], dtype=np.int64),
    ]
    river_network = network(groups, bifurcates=True)
    field = np.array([0.0, 10.0, 20.0, 30.0, 40.0])

    result = ekh.downstream.array.percentile(river_network, field, p=0.5, return_type="masked")

    np.testing.assert_allclose(result, [20.0, 30.0, 30.0, 35.0, 40.0])


@pytest.mark.skipif(not RUST, reason="Rust unavailable")
def test_bifurcation_path_scales_linearly():
    """The bifurcating traversal must stay linear in the number of edges.

    Each source fans out to two distinct leaves, so accumulators stay tiny and
    the measured time reflects the grouping/bookkeeping cost rather than the
    metric merge cost. Quadrupling the edge count should quadruple the runtime
    (not square it), which guards against a super-linear regression in the
    grouped accumulation path.
    """

    def fanout_network(width):
        sources = np.repeat(np.arange(width, dtype=np.int64), 2)
        leaves = np.arange(width, 3 * width, dtype=np.int64)
        groups = [np.vstack((leaves, sources))]
        field = np.arange(3 * width, dtype=np.float64) % 101
        return network(groups, bifurcates=True), field

    def median_seconds(width, repeats=5):
        river_network, field = fanout_network(width)
        # Warm up thread pool / allocations before timing.
        result = ekh.upstream.array.percentile(river_network, field, p=0.5, return_type="masked")
        # Each leaf is the median (= mean) of its source value and its own value.
        leaves = np.arange(width, 3 * width)
        source_of_leaf = (leaves - width) // 2
        expected = (field[source_of_leaf] + field[leaves]) / 2
        np.testing.assert_allclose(result[width:], expected)

        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            ekh.upstream.array.percentile(river_network, field, p=0.5, return_type="masked")
            samples.append(time.perf_counter() - start)
        return sorted(samples)[len(samples) // 2]

    base = median_seconds(50_000)
    quad = median_seconds(200_000)

    # Linear scaling gives ~4x; allow generous headroom for noise while still
    # catching quadratic behaviour (which would be ~16x).
    assert quad < base * 10
