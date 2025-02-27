import numpy as np
from conftest import *
from helper import read_network
from pytest_cases import parametrize

import earthkit.hydro as ekh


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
    np.testing.assert_array_equal(
        accum, ekh.flow_downstream(network, field)[..., network.mask]
    )
    np.testing.assert_array_equal(
        ekh.flow_downstream(network, field)[..., ~network.mask],
        field[..., ~network.mask],
    )
