import numpy as np
import pytest
from test_inputs.accumulation import *
from test_inputs.readers import *

import earthkit.hydro as ekh


@pytest.mark.parametrize(
    "river_network, input_field, flow_downstream, mv",
    [
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1a,
            flow_downstream_sum_1a,
            mv_1a,
        ),
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1b,
            flow_downstream_sum_1b,
            mv_1b,
        ),
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1c,
            flow_downstream_sum_1c,
            mv_1c,
        ),
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1d,
            flow_downstream_sum_1d,
            mv_1d,
        ),
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1g,
            flow_downstream_sum_1g,
            mv_1g,
        ),
        (
            ("cama_nextxy", cama_nextxy_2),
            input_field_2a,
            flow_downstream_sum_2a,
            mv_2a,
        ),
        (
            ("cama_nextxy", cama_nextxy_2),
            input_field_2b,
            flow_downstream_sum_2b,
            mv_2b,
        ),
        (
            ("cama_nextxy", cama_nextxy_2),
            input_field_2g,
            flow_downstream_sum_2g,
            mv_2g,
        ),
    ],
    indirect=["river_network"],
)
def test_flow_downstream_sum(river_network, input_field, flow_downstream, mv):
    output_field = ekh.flow_downstream(
        river_network,
        input_field,
        mv,
        in_place=False,
        ufunc=np.add,
        accept_missing=True,
    )
    print(output_field)
    print(flow_downstream)
    assert output_field.dtype == flow_downstream.dtype
    np.testing.assert_allclose(output_field, flow_downstream, atol=1e-10)


@pytest.mark.parametrize(
    "river_network, input_field, flow_downstream, mv",
    [
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1a,
            flow_downstream_max_1a,
            mv_1a,
        ),
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1b,
            flow_downstream_max_1b,
            mv_1b,
        ),
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1c,
            flow_downstream_max_1c,
            mv_1c,
        ),
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1d,
            flow_downstream_max_1d,
            mv_1d,
        ),
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1g,
            flow_downstream_max_1g,
            mv_1g,
        ),
    ],
    indirect=["river_network"],
)
def test_flow_downstream_max(river_network, input_field, flow_downstream, mv):
    output_field = ekh.flow_downstream(
        river_network,
        input_field,
        mv,
        in_place=False,
        ufunc=np.maximum,
        accept_missing=True,
    )
    print(output_field)
    print(flow_downstream)
    assert output_field.dtype == flow_downstream.dtype
    np.testing.assert_allclose(output_field, flow_downstream, atol=1e-10)


@pytest.mark.parametrize(
    "river_network, input_field, flow_downstream, mv",
    [
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1a,
            flow_downstream_min_1a,
            mv_1a,
        ),
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1b,
            flow_downstream_min_1b,
            mv_1b,
        ),
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1c,
            flow_downstream_min_1c,
            mv_1c,
        ),
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1d,
            flow_downstream_min_1d,
            mv_1d,
        ),
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1g,
            flow_downstream_min_1g,
            mv_1g,
        ),
    ],
    indirect=["river_network"],
)
def test_flow_downstream_min(river_network, input_field, flow_downstream, mv):
    output_field = ekh.flow_downstream(
        river_network,
        input_field,
        mv,
        in_place=False,
        ufunc=np.minimum,
        accept_missing=True,
    )
    print(output_field)
    print(flow_downstream)
    assert output_field.dtype == flow_downstream.dtype
    np.testing.assert_allclose(output_field, flow_downstream, atol=1e-10)


@pytest.mark.parametrize(
    "river_network, input_field, accum_field",
    [
        (("d8_ldd", d8_ldd_1), input_field_1b, flow_downstream_sum_1g),
        (("cama_downxy", cama_downxy_1), input_field_1b, flow_downstream_sum_1g),
        (("cama_nextxy", cama_nextxy_1), input_field_1b, flow_downstream_sum_1g),
        (("d8_ldd", d8_ldd_2), input_field_2b, flow_downstream_sum_2g),
        (("cama_downxy", cama_downxy_2), input_field_2b, flow_downstream_sum_2g),
        (("cama_nextxy", cama_nextxy_2), input_field_2b, flow_downstream_sum_2g),
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
