import numpy as np
import earthkit.hydro as ekh
from pytest_cases import parametrize
from conftest import *


def read_network(reader, map_name):
    if "d8_ldd" in reader:
        network = ekh.from_d8(map_name)
    elif "cama_downxy" in reader:
        network = ekh.from_cama_downxy(*map_name)
    elif "cama_nextxy" in reader:
        network = ekh.from_cama_nextxy(*map_name)
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
def test_accumulate_downstream(reader, map_name, upstream_points, N):
    network = read_network(reader, map_name)
    extra_dims = [np.random.randint(10) for _ in range(N)]
    field = np.ones((*extra_dims, network.n_nodes), dtype=int)
    accum = ekh.flow_downstream(network, field)
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
def test_accumulate_downstream_missing(reader, map_name, input_field, accum_field):
    network = read_network(reader, map_name)
    accum = ekh.flow_downstream(network, input_field, mv=-1, accept_missing=True)
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
def test_accumulate_downstream_2d(reader, map_name, N):
    network = read_network(reader, map_name)
    field = np.random.rand(*([np.random.randint(10)] * N), *network.mask.shape)
    field_1d = field[..., network.mask]
    accum = ekh.flow_downstream(network, field_1d)
    np.testing.assert_array_equal(accum, ekh.accumulate_downstream(network, field)[..., network.mask])
    np.testing.assert_array_equal(
        ekh.flow_downstream(network, field)[..., ~network.mask], field[..., ~network.mask]
    )


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
    ups = ekh.move_downstream(network, field)
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
    down = ekh.move_upstream(network, field)
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
def test_find_subcatchments_does_not_overwrite(reader, map_name):
    network = read_network(reader, map_name)
    field = np.arange(network.n_nodes) + 1
    subcatchments = ekh.find_subcatchments(network, field)
    print(subcatchments)
    print(field)
    np.testing.assert_array_equal(subcatchments, field)


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
def test_find_subcatchments_does_not_overwrite_2d(reader, map_name):
    network = read_network(reader, map_name)
    field = np.zeros(network.mask.shape, dtype="int")
    field[network.mask] = np.arange(network.n_nodes) + 1
    find_subcatchments = ekh.find_subcatchments(network, field)
    print(find_subcatchments)
    print(field)
    np.testing.assert_array_equal(find_subcatchments, field)


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
def test_find_subcatchments(reader, map_name, query_field, subcatchment):
    network = read_network(reader, map_name)
    subcatchments = ekh.find_subcatchments(network, query_field)
    print(subcatchment)
    print(subcatchments)
    np.testing.assert_array_equal(subcatchment, subcatchments)


@parametrize(
    "reader,map_name,query_field,find_subcatchments",
    [
        ("d8_ldd", d8_ldd_1, catchment_query_field_1, subcatchment_1),
        ("cama_downxy", cama_downxy_1, catchment_query_field_1, subcatchment_1),
        ("cama_nextxy", cama_nextxy_1, catchment_query_field_1, subcatchment_1),
        ("d8_ldd", d8_ldd_2, catchment_query_field_2, subcatchment_2),
        ("cama_downxy", cama_downxy_2, catchment_query_field_2, subcatchment_2),
        ("cama_nextxy", cama_nextxy_2, catchment_query_field_2, subcatchment_2),
    ],
)
def test_find_subcatchments_2d(reader, map_name, query_field, find_subcatchments):
    network = read_network(reader, map_name)
    field = np.zeros(network.mask.shape, dtype="int")
    field[network.mask] = query_field
    network_find_subcatchments = ekh.find_subcatchments(network, field)
    print(find_subcatchments)
    print(network_find_subcatchments)
    np.testing.assert_array_equal(network_find_subcatchments[network.mask], find_subcatchments)
    np.testing.assert_array_equal(network_find_subcatchments[~network.mask], 0)


@parametrize(
    "reader,map_name,query_field,find_catchments",
    [
        ("d8_ldd", d8_ldd_1, catchment_query_field_1, catchment_1),
        ("cama_downxy", cama_downxy_1, catchment_query_field_1, catchment_1),
        ("cama_nextxy", cama_nextxy_1, catchment_query_field_1, catchment_1),
        ("d8_ldd", d8_ldd_2, catchment_query_field_2, catchment_2),
        ("cama_downxy", cama_downxy_2, catchment_query_field_2, catchment_2),
        ("cama_nextxy", cama_nextxy_2, catchment_query_field_2, catchment_2),
    ],
)
def test_find_catchments(reader, map_name, query_field, find_catchments):
    network = read_network(reader, map_name)
    network_find_catchments = ekh.find_catchments(network, query_field)
    print(find_catchments)
    print(network_find_catchments)
    np.testing.assert_array_equal(network_find_catchments, find_catchments)


@parametrize(
    "reader,map_name,query_field,find_catchments",
    [
        ("d8_ldd", d8_ldd_1, catchment_query_field_1, catchment_1),
        ("cama_downxy", cama_downxy_1, catchment_query_field_1, catchment_1),
        ("cama_nextxy", cama_nextxy_1, catchment_query_field_1, catchment_1),
        ("d8_ldd", d8_ldd_2, catchment_query_field_2, catchment_2),
        ("cama_downxy", cama_downxy_2, catchment_query_field_2, catchment_2),
        ("cama_nextxy", cama_nextxy_2, catchment_query_field_2, catchment_2),
    ],
)
def test_find_catchments_2d(reader, map_name, query_field, find_catchments):
    network = read_network(reader, map_name)
    field = np.zeros(network.mask.shape, dtype="int")
    field[network.mask] = query_field
    network_find_catchments = ekh.find_catchments(network, field)
    print(find_catchments)
    print(network_find_catchments)
    np.testing.assert_array_equal(network_find_catchments[network.mask], find_catchments)
    np.testing.assert_array_equal(network_find_catchments[~network.mask], 0)


@parametrize(
    "reader,map_name,mask,accumulate_downstream",
    [
        ("d8_ldd", d8_ldd_2, mask_2, masked_unit_accuflux_2),
        ("cama_downxy", cama_downxy_2, mask_2, masked_unit_accuflux_2),
        ("cama_nextxy", cama_nextxy_2, mask_2, masked_unit_accuflux_2),
    ],
)
def test_subnetwork(reader, map_name, mask, accumulate_downstream):
    network = read_network(reader, map_name)
    network = network.create_subnetwork(mask)
    field = np.ones(network.n_nodes)
    accum = ekh.accumulate_downstream(network, field)
    print(accum)
    print(accumulate_downstream)
    np.testing.assert_array_equal(accum, accumulate_downstream)
