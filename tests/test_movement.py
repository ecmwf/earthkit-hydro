import numpy as np
from conftest import *
from helper import read_network
from pytest_cases import parametrize

import earthkit.hydro as ekh


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
