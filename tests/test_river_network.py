import numpy as np
from earthkit.hydro import from_d8, from_cama_downxy, from_cama_nextxy

# import pytest
from pytest_cases import parametrize
from conftest import *


def read_network(reader, map_name):
    if "d8_ldd" in reader:
        network = from_d8(map_name)
    elif "cama_downxy" in reader:
        network = from_cama_downxy(*map_name)
    elif "cama_nextxy" in reader:
        network = from_cama_nextxy(*map_name)
    else:
        raise Exception("Unknown map type")
    return network


@parametrize(
    "reader,map_name,upstream_points",
    [
        ("d8_ldd", d8_ldd_1, unit_field_accuflux_1),
        ("cama_downxy", cama_downxy_1, unit_field_accuflux_1),
        ("cama_nextxy", cama_nextxy_1, unit_field_accuflux_1),
        ("d8_ldd", d8_ldd_2, unit_field_accuflux_2),
        ("cama_downxy", cama_downxy_2, unit_field_accuflux_2),
        ("cama_nextxy", cama_nextxy_2, unit_field_accuflux_2),
    ],
)
@parametrize("N", range(4))
def test_accuflux(reader, map_name, upstream_points, N):
    network = read_network(reader, map_name)
    extra_dims = [np.random.randint(10) for _ in range(N)]
    field = np.ones((*extra_dims, network.n_nodes), dtype=int)
    accum = network.accuflux(field)
    print(accum[..., :])
    print(upstream_points)
    extended_upstream_points = np.tile(upstream_points, extra_dims + [1])
    np.testing.assert_array_equal(accum, extended_upstream_points)


@parametrize(
    "reader,map_name,input_field,accum_field",
    [
        ("d8_ldd", d8_ldd_1, input_field_accuflux_1, field_accuflux_1),
        ("cama_downxy", cama_downxy_1, input_field_accuflux_1, field_accuflux_1),
        ("cama_nextxy", cama_nextxy_1, input_field_accuflux_1, field_accuflux_1),
        ("d8_ldd", d8_ldd_2, input_field_accuflux_2, field_accuflux_2),
        ("cama_downxy", cama_downxy_2, input_field_accuflux_2, field_accuflux_2),
        ("cama_nextxy", cama_nextxy_2, input_field_accuflux_2, field_accuflux_2),
    ],
)
def test_accuflux_missing(reader, map_name, input_field, accum_field):
    network = read_network(reader, map_name)
    accum = network.accuflux(input_field, mv=-1, accept_missing=True)
    print(accum)
    print(accum_field)
    np.testing.assert_array_equal(accum, accum_field)


@parametrize(
    "reader,map_name",
    [
        ("d8_ldd", d8_ldd_1),
        ("cama_downxy", cama_downxy_1),
        ("cama_nextxy", cama_nextxy_1),
        ("d8_ldd", d8_ldd_2),
        ("cama_downxy", cama_downxy_2),
        ("cama_nextxy", cama_nextxy_2),
    ],
)
@parametrize("N", range(4))
def test_accuflux_2d(reader, map_name, N):
    network = read_network(reader, map_name)
    field = np.random.rand(*([np.random.randint(10)] * N), *network.mask.shape)
    field_1d = field[..., network.mask]
    accum = network.accuflux(field_1d)
    np.testing.assert_array_equal(accum, network.accuflux(field)[..., network.mask])
    np.testing.assert_array_equal(network.accuflux(field)[..., ~network.mask], field[..., ~network.mask])


@parametrize(
    "reader,map_name,downstream_nodes",
    [
        ("d8_ldd", d8_ldd_1, downstream_nodes_1),
        ("cama_downxy", cama_downxy_1, downstream_nodes_1),
        ("cama_nextxy", cama_nextxy_1, downstream_nodes_1),
        ("d8_ldd", d8_ldd_2, downstream_nodes_2),
        ("cama_downxy", cama_downxy_2, downstream_nodes_2),
        ("cama_nextxy", cama_nextxy_2, downstream_nodes_2),
    ],
)
def test_downstream_nodes(reader, map_name, downstream_nodes):
    network = read_network(reader, map_name)
    print(network.downstream_nodes)
    print(downstream_nodes)
    np.testing.assert_array_equal(network.downstream_nodes, downstream_nodes)


@parametrize(
    "reader,map_name,upstream",
    [
        ("d8_ldd", d8_ldd_1, upstream_1),
        ("cama_downxy", cama_downxy_1, upstream_1),
        ("cama_nextxy", cama_nextxy_1, upstream_1),
        ("d8_ldd", d8_ldd_2, upstream_2),
        ("cama_downxy", cama_downxy_2, upstream_2),
        ("cama_nextxy", cama_nextxy_2, upstream_2),
    ],
)
def test_upstream(reader, map_name, upstream):
    network = read_network(reader, map_name)
    field = np.arange(1, network.n_nodes + 1)
    ups = network.upstream(field)
    np.testing.assert_array_equal(ups, upstream)


@parametrize(
    "reader,map_name,downstream",
    [
        ("d8_ldd", d8_ldd_1, downstream_1),
        ("cama_downxy", cama_downxy_1, downstream_1),
        ("cama_nextxy", cama_nextxy_1, downstream_1),
        ("d8_ldd", d8_ldd_2, downstream_2),
        ("cama_downxy", cama_downxy_2, downstream_2),
        ("cama_nextxy", cama_nextxy_2, downstream_2),
    ],
)
def test_downstream(reader, map_name, downstream):
    network = read_network(reader, map_name)
    field = np.arange(1, network.n_nodes + 1)
    down = network.downstream(field)
    print(down)
    print(downstream)
    np.testing.assert_array_equal(down, downstream)


@parametrize(
    "reader,map_name",
    [
        ("d8_ldd", d8_ldd_1),
        ("cama_downxy", cama_downxy_1),
        ("cama_nextxy", cama_nextxy_1),
        ("d8_ldd", d8_ldd_2),
        ("cama_downxy", cama_downxy_2),
        ("cama_nextxy", cama_nextxy_2),
    ],
)
def test_subcatchment_does_not_overwrite(reader, map_name):
    network = read_network(reader, map_name)
    field = np.arange(network.n_nodes) + 1
    subcatchment = network.subcatchment(field)
    print(subcatchment)
    print(field)
    np.testing.assert_array_equal(subcatchment, field)


@parametrize(
    "reader,map_name",
    [
        ("d8_ldd", d8_ldd_1),
        ("cama_downxy", cama_downxy_1),
        ("cama_nextxy", cama_nextxy_1),
        ("d8_ldd", d8_ldd_2),
        ("cama_downxy", cama_downxy_2),
        ("cama_nextxy", cama_nextxy_2),
    ],
)
def test_subcatchment_does_not_overwrite_2d(reader, map_name):
    network = read_network(reader, map_name)
    field = np.zeros(network.mask.shape, dtype="int")
    field[network.mask] = np.arange(network.n_nodes) + 1
    subcatchment = network.subcatchment(field)
    print(subcatchment)
    print(field)
    np.testing.assert_array_equal(subcatchment, field)


@parametrize(
    "reader,map_name,query_field,subcatchment",
    [
        ("d8_ldd", d8_ldd_1, catchment_query_field_1, subcatchment_1),
        ("cama_downxy", cama_downxy_1, catchment_query_field_1, subcatchment_1),
        ("cama_nextxy", cama_nextxy_1, catchment_query_field_1, subcatchment_1),
        ("d8_ldd", d8_ldd_2, catchment_query_field_2, subcatchment_2),
        ("cama_downxy", cama_downxy_2, catchment_query_field_2, subcatchment_2),
        ("cama_nextxy", cama_nextxy_2, catchment_query_field_2, subcatchment_2),
    ],
)
def test_subcatchment(reader, map_name, query_field, subcatchment):
    network = read_network(reader, map_name)
    network_subcatchment = network.subcatchment(query_field)
    print(subcatchment)
    print(network_subcatchment)
    np.testing.assert_array_equal(network_subcatchment, subcatchment)


@parametrize(
    "reader,map_name,query_field,subcatchment",
    [
        ("d8_ldd", d8_ldd_1, catchment_query_field_1, subcatchment_1),
        ("cama_downxy", cama_downxy_1, catchment_query_field_1, subcatchment_1),
        ("cama_nextxy", cama_nextxy_1, catchment_query_field_1, subcatchment_1),
        ("d8_ldd", d8_ldd_2, catchment_query_field_2, subcatchment_2),
        ("cama_downxy", cama_downxy_2, catchment_query_field_2, subcatchment_2),
        ("cama_nextxy", cama_nextxy_2, catchment_query_field_2, subcatchment_2),
    ],
)
def test_subcatchment_2d(reader, map_name, query_field, subcatchment):
    network = read_network(reader, map_name)
    field = np.zeros(network.mask.shape, dtype="int")
    field[network.mask] = query_field
    network_subcatchment = network.subcatchment(field)
    print(subcatchment)
    print(network_subcatchment)
    np.testing.assert_array_equal(network_subcatchment[network.mask], subcatchment)
    np.testing.assert_array_equal(network_subcatchment[~network.mask], 0)


@parametrize(
    "reader,map_name,query_field,catchment",
    [
        ("d8_ldd", d8_ldd_1, catchment_query_field_1, catchment_1),
        ("cama_downxy", cama_downxy_1, catchment_query_field_1, catchment_1),
        ("cama_nextxy", cama_nextxy_1, catchment_query_field_1, catchment_1),
        ("d8_ldd", d8_ldd_2, catchment_query_field_2, catchment_2),
        ("cama_downxy", cama_downxy_2, catchment_query_field_2, catchment_2),
        ("cama_nextxy", cama_nextxy_2, catchment_query_field_2, catchment_2),
    ],
)
def test_catchment(reader, map_name, query_field, catchment):
    network = read_network(reader, map_name)
    network_catchment = network.catchment(query_field)
    print(catchment)
    print(network_catchment)
    np.testing.assert_array_equal(network_catchment, catchment)


@parametrize(
    "reader,map_name,query_field,catchment",
    [
        ("d8_ldd", d8_ldd_1, catchment_query_field_1, catchment_1),
        ("cama_downxy", cama_downxy_1, catchment_query_field_1, catchment_1),
        ("cama_nextxy", cama_nextxy_1, catchment_query_field_1, catchment_1),
        ("d8_ldd", d8_ldd_2, catchment_query_field_2, catchment_2),
        ("cama_downxy", cama_downxy_2, catchment_query_field_2, catchment_2),
        ("cama_nextxy", cama_nextxy_2, catchment_query_field_2, catchment_2),
    ],
)
def test_catchment_2d(reader, map_name, query_field, catchment):
    network = read_network(reader, map_name)
    field = np.zeros(network.mask.shape, dtype="int")
    field[network.mask] = query_field
    network_catchment = network.catchment(field)
    print(catchment)
    print(network_catchment)
    np.testing.assert_array_equal(network_catchment[network.mask], catchment)
    np.testing.assert_array_equal(network_catchment[~network.mask], 0)


@parametrize(
    "reader,map_name,mask,accuflux",
    [
        ("d8_ldd", d8_ldd_2, mask_2, masked_unit_accuflux_2),
        ("cama_downxy", cama_downxy_2, mask_2, masked_unit_accuflux_2),
        ("cama_nextxy", cama_nextxy_2, mask_2, masked_unit_accuflux_2),
    ],
)
def test_subnetwork(reader, map_name, mask, accuflux):
    network = read_network(reader, map_name)
    network = network.create_subnetwork(mask)
    field = np.ones(network.n_nodes)
    accum = network.accuflux(field)
    print(accum)
    print(accuflux)
    np.testing.assert_array_equal(accum, accuflux)
