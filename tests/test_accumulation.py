import numpy as np
import pytest
from test_inputs.accumulation import *
from test_inputs.readers import *

import earthkit.hydro as ekh


@pytest.mark.parametrize(
    "river_network, upstream_points",
    [
        (("d8_ldd", d8_ldd_1), unit_field_accuflux_1),
        (("cama_downxy", cama_downxy_1), unit_field_accuflux_1),
        (("cama_nextxy", cama_nextxy_1), unit_field_accuflux_1),
        (("d8_ldd", d8_ldd_2), unit_field_accuflux_2),
        (("cama_downxy", cama_downxy_2), unit_field_accuflux_2),
        (("cama_nextxy", cama_nextxy_2), unit_field_accuflux_2),
    ],
    indirect=["river_network"],
)
@pytest.mark.parametrize("N", range(4))
def test_accumulate_downstream(river_network, upstream_points, N):
    extra_dims = [np.random.randint(10) for _ in range(N)]
    field = np.ones((*extra_dims, river_network.n_nodes), dtype=int)
    accum = ekh.flow_downstream(river_network, field)
    print(accum[..., :])
    print(upstream_points)
    extended_upstream_points = np.tile(upstream_points, extra_dims + [1])
    np.testing.assert_array_equal(accum, extended_upstream_points)


@pytest.mark.parametrize(
    "river_network, input_field, accum_field",
    [
        (("d8_ldd", d8_ldd_1), input_field_accuflux_1, field_accuflux_1),
        (("cama_downxy", cama_downxy_1), input_field_accuflux_1, field_accuflux_1),
        (("cama_nextxy", cama_nextxy_1), input_field_accuflux_1, field_accuflux_1),
        (("d8_ldd", d8_ldd_2), input_field_accuflux_2, field_accuflux_2),
        (("cama_downxy", cama_downxy_2), input_field_accuflux_2, field_accuflux_2),
        (("cama_nextxy", cama_nextxy_2), input_field_accuflux_2, field_accuflux_2),
    ],
    indirect=["river_network"],
)
def test_accumulate_downstream_missing(river_network, input_field, accum_field):
    accum = ekh.flow_downstream(river_network, input_field, mv=-1, accept_missing=True)
    print(accum)
    print(accum_field)
    np.testing.assert_array_equal(accum, accum_field)


@pytest.mark.parametrize(
    "river_network",
    [
        ("d8_ldd", d8_ldd_1),
        ("cama_downxy", cama_downxy_1),
        ("cama_nextxy", cama_nextxy_1),
        ("d8_ldd", d8_ldd_2),
        ("cama_downxy", cama_downxy_2),
        ("cama_nextxy", cama_nextxy_2),
    ],
    indirect=True,
)
@pytest.mark.parametrize("N", range(4))
def test_accumulate_downstream_2d(river_network, N):
    field = np.random.rand(*([np.random.randint(10)] * N), *river_network.mask.shape)
    field_1d = field[..., river_network.mask]
    accum = ekh.flow_downstream(river_network, field_1d)
    np.testing.assert_array_equal(
        accum, ekh.flow_downstream(river_network, field)[..., river_network.mask]
    )
    np.testing.assert_array_equal(
        ekh.flow_downstream(river_network, field)[..., ~river_network.mask],
        field[..., ~river_network.mask],
    )
