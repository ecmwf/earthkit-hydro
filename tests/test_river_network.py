import numpy as np
from earthkit.hydro import from_d8, from_cama_downxy
# import pytest
from pytest_cases import parametrize
from conftest import *

@parametrize("reader,map_name,upstream_points",
            [('d8_ldd',d8_ldd_1,unit_field_accuflux_1),
            ('cama_downxy',cama_downxy_1,unit_field_accuflux_1),
            ('d8_ldd',d8_ldd_2,unit_field_accuflux_2),
            ('cama_downxy',cama_downxy_2,unit_field_accuflux_2),
            ])
@parametrize("N", range(4))
def test_accuflux(reader, map_name, upstream_points, N):
    if 'd8_ldd' in reader:
        network = from_d8(map_name)
    elif 'cama_downxy' in reader:
        network = from_cama_downxy(*map_name)
    else:
        raise Exception("Unknown map type")
    extra_dims = [np.random.randint(10) for _ in range(N)]
    field = np.ones((*extra_dims, network.n_nodes), dtype=int)
    accum = network.accuflux(field)
    print(accum[...,:])
    print(upstream_points)
    extended_upstream_points = np.tile(upstream_points, extra_dims+[1])
    np.testing.assert_array_equal(accum, extended_upstream_points)


@parametrize("reader,map_name",
            [('d8_ldd',d8_ldd_1),
            ('cama_downxy',cama_downxy_1),
            ('d8_ldd',d8_ldd_2),
            ('cama_downxy',cama_downxy_2),
            ])
@parametrize("N", range(4))
def test_accuflux_2d(reader, map_name, N):
    if 'd8_ldd' in reader:
        network = from_d8(map_name)
    elif 'cama_downxy' in reader:
        network = from_cama_downxy(*map_name)
    else:
        raise Exception("Unknown map type")
    field = np.random.rand(*([np.random.randint(10)]*N), *network.mask.shape)
    field_1d = field[...,network.mask]
    accum = network.accuflux(field_1d)
    np.testing.assert_array_equal(accum, network.accuflux(field)[...,network.mask])

@parametrize("reader,map_name,downstream_nodes",
            [('d8_ldd',d8_ldd_1,downstream_nodes_1),
            ('cama_downxy',cama_downxy_1,downstream_nodes_1),
            ('d8_ldd',d8_ldd_2,downstream_nodes_2),
            ('cama_downxy',cama_downxy_2,downstream_nodes_2),
            ])
def test_downstream_nodes(reader, map_name, downstream_nodes):
    if 'd8_ldd' in reader:
        network = from_d8(map_name)
    elif 'cama_downxy' in reader:
        network = from_cama_downxy(*map_name)
    else:
        raise Exception("Unknown map type")
    print(network.downstream_nodes)
    print(downstream_nodes)
    np.testing.assert_array_equal(network.downstream_nodes, downstream_nodes)

@parametrize("reader,map_name,upstream",
            [('d8_ldd',d8_ldd_1,upstream_1),
            ('cama_downxy',cama_downxy_1,upstream_1),
            ('d8_ldd',d8_ldd_2,upstream_2),
            ('cama_downxy',cama_downxy_2,upstream_2),
            ])
def test_upstream(reader, map_name, upstream):
    if 'd8_ldd' in reader:
        network = from_d8(map_name)
    elif 'cama_downxy' in reader:
        network = from_cama_downxy(*map_name)
    else:
        raise Exception("Unknown map type")
    field = np.arange(1, network.n_nodes+1)
    ups = network.upstream(field)
    np.testing.assert_array_equal(ups, upstream)

@parametrize("reader,map_name,downstream",
            [('d8_ldd',d8_ldd_1,downstream_1),
            ('cama_downxy',cama_downxy_1,downstream_1),
            ('d8_ldd',d8_ldd_2,downstream_2),
            ('cama_downxy',cama_downxy_2,downstream_2),
            ])
def test_downstream(reader, map_name, downstream):
    if 'd8_ldd' in reader:
        network = from_d8(map_name)
    elif 'cama_downxy' in reader:
        network = from_cama_downxy(*map_name)
    else:
        raise Exception("Unknown map type")
    field = np.arange(1, network.n_nodes+1)
    down = network.downstream(field)
    print(down)
    print(downstream)
    np.testing.assert_array_equal(down, downstream)

