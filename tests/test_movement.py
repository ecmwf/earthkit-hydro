import numpy as np
import pytest
from test_inputs.movement import *
from test_inputs.readers import *

import earthkit.hydro as ekh


@pytest.mark.parametrize(
    "river_network, upstream",
    [
        (("d8_ldd", d8_ldd_1), upstream_1),
        (("cama_downxy", cama_downxy_1), upstream_1),
        (("cama_nextxy", cama_nextxy_1), upstream_1),
        (("d8_ldd", d8_ldd_2), upstream_2),
        (("cama_downxy", cama_downxy_2), upstream_2),
        (("cama_nextxy", cama_nextxy_2), upstream_2),
    ],
    indirect=["river_network"],
)
def test_upstream(river_network, upstream):
    field = np.arange(1, river_network.n_nodes + 1)
    ups = ekh.move_downstream(river_network, field)
    np.testing.assert_array_equal(ups, upstream)


@pytest.mark.parametrize(
    "river_network, downstream",
    [
        (("d8_ldd", d8_ldd_1), downstream_1),
        (("cama_downxy", cama_downxy_1), downstream_1),
        (("cama_nextxy", cama_nextxy_1), downstream_1),
        (("d8_ldd", d8_ldd_2), downstream_2),
        (("cama_downxy", cama_downxy_2), downstream_2),
        (("cama_nextxy", cama_nextxy_2), downstream_2),
    ],
    indirect=["river_network"],
)
def test_downstream(river_network, downstream):
    field = np.arange(1, river_network.n_nodes + 1)
    down = ekh.move_upstream(river_network, field)
    print(down)
    print(downstream)
    np.testing.assert_array_equal(down, downstream)


@pytest.mark.parametrize(
    "river_network, upstream",
    [
        (("d8_ldd", d8_ldd_1), upstream_1),
        (("cama_downxy", cama_downxy_1), upstream_1),
        (("cama_nextxy", cama_nextxy_1), upstream_1),
        (("d8_ldd", d8_ldd_2), upstream_2),
        (("cama_downxy", cama_downxy_2), upstream_2),
        (("cama_nextxy", cama_nextxy_2), upstream_2),
    ],
    indirect=["river_network"],
)
def test_upstream_ND(river_network, upstream):
    field = np.arange(1, river_network.n_nodes + 1)
    field = np.stack([field, field], axis=0)
    ups = ekh.move_downstream(river_network, field)
    np.testing.assert_array_equal(ups, np.stack([upstream, upstream], axis=0))


@pytest.mark.parametrize(
    "river_network, downstream",
    [
        (("d8_ldd", d8_ldd_1), downstream_1),
        (("cama_downxy", cama_downxy_1), downstream_1),
        (("cama_nextxy", cama_nextxy_1), downstream_1),
        (("d8_ldd", d8_ldd_2), downstream_2),
        (("cama_downxy", cama_downxy_2), downstream_2),
        (("cama_nextxy", cama_nextxy_2), downstream_2),
    ],
    indirect=["river_network"],
)
def test_downstream_ND(river_network, downstream):
    field = np.arange(1, river_network.n_nodes + 1)
    field = np.stack([field, field], axis=0)
    print(field.shape)
    ups = ekh.move_upstream(river_network, field)
    np.testing.assert_array_equal(ups, np.stack([downstream, downstream], axis=0))
