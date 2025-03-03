import numpy as np
import pytest
from test_inputs.accumulation import *
from test_inputs.readers import *

import earthkit.hydro as ekh


@pytest.mark.parametrize(
    "river_network, input_field, flow_downstream, metric, mv",
    [
        (("d8_ldd", d8_ldd_1), input_field_1a, flow_downstream_sum_1a, np.add, mv_1a),
        (
            ("d8_ldd", d8_ldd_1),
            input_field_1a,
            flow_downstream_max_1a,
            np.maximum,
            mv_1a,
        ),
        (
            ("d8_ldd", d8_ldd_1),
            input_field_1a,
            flow_downstream_min_1a,
            np.minimum,
            mv_1a,
        ),
        (
            ("cama_downxy", cama_downxy_1),
            input_field_1a,
            flow_downstream_sum_1a,
            np.add,
            mv_1a,
        ),
        (
            ("cama_downxy", cama_downxy_1),
            input_field_1a,
            flow_downstream_max_1a,
            np.maximum,
            mv_1a,
        ),
        (
            ("cama_downxy", cama_downxy_1),
            input_field_1a,
            flow_downstream_min_1a,
            np.minimum,
            mv_1a,
        ),
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1a,
            flow_downstream_sum_1a,
            np.add,
            mv_1a,
        ),
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1a,
            flow_downstream_max_1a,
            np.maximum,
            mv_1a,
        ),
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1a,
            flow_downstream_min_1a,
            np.minimum,
            mv_1a,
        ),
        (("d8_ldd", d8_ldd_1), input_field_1b, flow_downstream_sum_1b, np.add, mv_1b),
        (
            ("d8_ldd", d8_ldd_1),
            input_field_1b,
            flow_downstream_max_1b,
            np.maximum,
            mv_1b,
        ),
        (
            ("d8_ldd", d8_ldd_1),
            input_field_1b,
            flow_downstream_min_1b,
            np.minimum,
            mv_1b,
        ),
        (
            ("cama_downxy", cama_downxy_1),
            input_field_1b,
            flow_downstream_sum_1b,
            np.add,
            mv_1b,
        ),
        (
            ("cama_downxy", cama_downxy_1),
            input_field_1b,
            flow_downstream_max_1b,
            np.maximum,
            mv_1b,
        ),
        (
            ("cama_downxy", cama_downxy_1),
            input_field_1b,
            flow_downstream_min_1b,
            np.minimum,
            mv_1b,
        ),
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1b,
            flow_downstream_sum_1b,
            np.add,
            mv_1b,
        ),
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1b,
            flow_downstream_max_1b,
            np.maximum,
            mv_1b,
        ),
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1b,
            flow_downstream_min_1b,
            np.minimum,
            mv_1b,
        ),
        (("d8_ldd", d8_ldd_1), input_field_1g, flow_downstream_sum_1g, np.add, mv_1g),
        (
            ("d8_ldd", d8_ldd_1),
            input_field_1g,
            flow_downstream_max_1g,
            np.maximum,
            mv_1g,
        ),
        (
            ("d8_ldd", d8_ldd_1),
            input_field_1g,
            flow_downstream_min_1g,
            np.minimum,
            mv_1g,
        ),
        (
            ("cama_downxy", cama_downxy_1),
            input_field_1g,
            flow_downstream_sum_1g,
            np.add,
            mv_1g,
        ),
        (
            ("cama_downxy", cama_downxy_1),
            input_field_1g,
            flow_downstream_max_1g,
            np.maximum,
            mv_1g,
        ),
        (
            ("cama_downxy", cama_downxy_1),
            input_field_1g,
            flow_downstream_min_1g,
            np.minimum,
            mv_1g,
        ),
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1g,
            flow_downstream_sum_1g,
            np.add,
            mv_1g,
        ),
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1g,
            flow_downstream_max_1g,
            np.maximum,
            mv_1g,
        ),
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1g,
            flow_downstream_min_1g,
            np.minimum,
            mv_1g,
        ),
    ],
    indirect=["river_network"],
)
@pytest.mark.parametrize("N", range(4))
def test_flow_downstream(river_network, input_field, flow_downstream, metric, mv, N):
    print("input", input_field)
    output_field = ekh.flow_downstream(
        river_network,
        input_field,
        mv,
        in_place=False,
        ufunc=metric,
        accept_missing=True,
    )
    print("output", output_field)
    print(flow_downstream)
    np.testing.assert_array_equal(output_field, flow_downstream)


# def test_flow_downstream(river_network, flow_downstream, metric, mv, N):
#     extra_dims = [np.random.randint(10) for _ in range(N)]
#     field = np.ones((*extra_dims, river_network.n_nodes), dtype=int)
#     output_field = ekh.flow_downstream(river_network, field)
#     print(accum[..., :])
#     print(upstream_points)
#     extended_upstream_points = np.tile(upstream_points, extra_dims + [1])
#     np.testing.assert_array_equal(accum, extended_upstream_points)


@pytest.mark.parametrize(
    "river_network, input_field, accum_field",
    [
        (("d8_ldd", d8_ldd_1), input_field_1b, flow_downstream_sum_1g),
        (("cama_downxy", cama_downxy_1), input_field_1b, flow_downstream_sum_1g),
        (("cama_nextxy", cama_nextxy_1), input_field_1b, flow_downstream_sum_1g),
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
