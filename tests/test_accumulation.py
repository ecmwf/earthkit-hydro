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
            upstream_metric_sum_1a,
            mv_1a,
        ),
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1b,
            upstream_metric_sum_1b,
            mv_1b,
        ),
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1c,
            upstream_metric_sum_1c,
            mv_1c,
        ),
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1d,
            upstream_metric_sum_1d,
            mv_1d,
        ),
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1g,
            upstream_metric_sum_1g,
            mv_1g,
        ),
        (
            ("cama_nextxy", cama_nextxy_2),
            input_field_2a,
            upstream_metric_sum_2a,
            mv_2a,
        ),
        (
            ("cama_nextxy", cama_nextxy_2),
            input_field_2b,
            upstream_metric_sum_2b,
            mv_2b,
        ),
        (
            ("cama_nextxy", cama_nextxy_2),
            input_field_2g,
            upstream_metric_sum_2g,
            mv_2g,
        ),
    ],
    indirect=["river_network"],
)
def test_upstream_metric_sum(river_network, input_field, flow_downstream, mv):
    output_field = ekh.calculate_upstream_metric(
        river_network, input_field, "sum", weights=None, mv=mv, accept_missing=True
    )
    print(output_field)
    print(flow_downstream)
    assert output_field.dtype == flow_downstream.dtype
    np.testing.assert_allclose(output_field, flow_downstream)
    np.testing.assert_allclose(
        output_field,
        ekh.flow_downstream(
            river_network,
            input_field,
            mv,
            in_place=False,
            ufunc=np.add,
            accept_missing=True,
        ),
    )


@pytest.mark.parametrize(
    "river_network, input_field, flow_downstream, mv",
    [
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1a,
            upstream_metric_max_1a,
            mv_1a,
        ),
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1b,
            upstream_metric_max_1b,
            mv_1b,
        ),
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1c,
            upstream_metric_max_1c,
            mv_1c,
        ),
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1d,
            upstream_metric_max_1d,
            mv_1d,
        ),
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1g,
            upstream_metric_max_1g,
            mv_1g,
        ),
    ],
    indirect=["river_network"],
)
def test_calculate_upstream_metric_max(river_network, input_field, flow_downstream, mv):
    output_field = ekh.calculate_upstream_metric(
        river_network, input_field, "max", weights=None, mv=mv, accept_missing=True
    )
    print(output_field)
    print(flow_downstream)
    assert output_field.dtype == flow_downstream.dtype
    np.testing.assert_allclose(output_field, flow_downstream)
    np.testing.assert_allclose(
        output_field,
        ekh.flow_downstream(
            river_network,
            input_field,
            mv,
            in_place=False,
            ufunc=np.maximum,
            accept_missing=True,
        ),
    )


@pytest.mark.parametrize(
    "river_network, input_field, flow_downstream, mv",
    [
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1a,
            upstream_metric_min_1a,
            mv_1a,
        ),
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1b,
            upstream_metric_min_1b,
            mv_1b,
        ),
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1c,
            upstream_metric_min_1c,
            mv_1c,
        ),
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1d,
            upstream_metric_min_1d,
            mv_1d,
        ),
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1g,
            upstream_metric_min_1g,
            mv_1g,
        ),
    ],
    indirect=["river_network"],
)
def test_calculate_upstream_metric_min(river_network, input_field, flow_downstream, mv):
    output_field = ekh.calculate_upstream_metric(
        river_network, input_field, "min", weights=None, mv=mv, accept_missing=True
    )
    print(output_field)
    print(flow_downstream)
    assert output_field.dtype == flow_downstream.dtype
    np.testing.assert_allclose(output_field, flow_downstream)
    np.testing.assert_allclose(
        output_field,
        ekh.flow_downstream(
            river_network,
            input_field,
            mv,
            in_place=False,
            ufunc=np.minimum,
            accept_missing=True,
        ),
    )


@pytest.mark.parametrize(
    "river_network, input_field, flow_downstream, mv",
    [
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1a,
            upstream_metric_mean_1a,
            mv_1a,
        ),
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1b,
            upstream_metric_mean_1b,
            mv_1b,
        ),
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1c,
            upstream_metric_mean_1c,
            mv_1c,
        ),
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1d,
            upstream_metric_mean_1d,
            mv_1d,
        ),
    ],
    indirect=["river_network"],
)
def test_calculate_upstream_metric_mean(
    river_network, input_field, flow_downstream, mv
):
    output_field = ekh.calculate_upstream_metric(
        river_network, input_field, "mean", weights=None, mv=mv, accept_missing=True
    )
    assert output_field.dtype == flow_downstream.dtype
    np.testing.assert_allclose(output_field, flow_downstream)


@pytest.mark.parametrize(
    "river_network, input_field, accum_field",
    [
        (("d8_ldd", d8_ldd_1), input_field_1b, upstream_metric_sum_1g),
        (("cama_downxy", cama_downxy_1), input_field_1b, upstream_metric_sum_1g),
        (("cama_nextxy", cama_nextxy_1), input_field_1b, upstream_metric_sum_1g),
        (("d8_ldd", d8_ldd_2), input_field_2b, upstream_metric_sum_2g),
        (("cama_downxy", cama_downxy_2), input_field_2b, upstream_metric_sum_2g),
        (("cama_nextxy", cama_nextxy_2), input_field_2b, upstream_metric_sum_2g),
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
