import numpy as np
from earthkit.hydro import from_d8


def test_accuflux(d8_ldd, upstream_points):
    network = from_d8(d8_ldd)
    field = np.ones(network.n_nodes, dtype=int)
    accum = network.accuflux(field)

    print(accum)
    print(upstream_points)
    np.testing.assert_array_equal(accum, upstream_points)


def test_downstream_nodes(d8_ldd, downstream_nodes):
    network = from_d8(d8_ldd)
    print(network.downstream_nodes)
    print(downstream_nodes)
    np.testing.assert_array_equal(network.downstream_nodes, downstream_nodes)


def test_upstream_points(d8_ldd, upstream_points):
    network = from_d8(d8_ldd)
    ups = network.upstream_points()
    np.testing.assert_array_equal(ups, upstream_points)


def test_upstream(d8_ldd, test_field, upstream):
    network = from_d8(d8_ldd)
    ups = network.upstream(test_field)
    print(ups)
    print(upstream)
    np.testing.assert_array_equal(ups, upstream)


def test_downstream(d8_ldd, test_field, downstream):
    network = from_d8(d8_ldd)
    down = network.downstream(test_field)
    print(down)
    print(downstream)
    np.testing.assert_array_equal(down, downstream)

# def test_catchments(d8_ldd):
#     network = from_d8(d8_ldd)
#     catchments = network.catchment(11)
#     assert False

