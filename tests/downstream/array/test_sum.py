import numpy as np
import pytest
from _test_inputs.accumulation import *
from _test_inputs.readers import *
from utils import convert_to_2d

import earthkit.hydro as ekh


@pytest.mark.parametrize(
    "river_network, input_field, flow_downstream, mv",
    [
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1c,
            downstream_metric_sum_1c,
            mv_1c,
        ),
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1e,
            downstream_metric_sum_1e,
            mv_1e,
        ),
    ],
    indirect=["river_network"],
)
@pytest.mark.parametrize("array_backend", ["numpy", "torch", "jax"])
def test_downstream_metric_sum(
    river_network, input_field, flow_downstream, mv, array_backend
):
    river_network = river_network.to_device("cpu", array_backend)
    xp = ekh._backends.find.get_array_backend(array_backend)
    output_field = ekh.downstream.array.sum(
        river_network, xp.asarray(input_field), node_weights=None, return_type="masked"
    )
    output_field = np.asarray(output_field)
    flow_downstream_out = np.asarray(xp.asarray(flow_downstream))
    print(output_field)
    print(flow_downstream_out)
    assert output_field.dtype == flow_downstream_out.dtype
    np.testing.assert_allclose(output_field, flow_downstream, rtol=1e-6, equal_nan=True)

    print(input_field)
    input_field = convert_to_2d(river_network, input_field, 0)
    flow_downstream = convert_to_2d(river_network, flow_downstream, 0)
    print(mv, input_field.dtype)
    print(input_field, flow_downstream)
    output_field = ekh.downstream.array.sum(
        river_network, xp.asarray(input_field), node_weights=None
    )
    output_field = np.asarray(output_field).flatten()
    flow_downstream = np.asarray(xp.asarray(flow_downstream))
    print(output_field)
    print(flow_downstream)
    assert output_field.dtype == flow_downstream.dtype
    np.testing.assert_allclose(output_field, flow_downstream, rtol=1e-6, equal_nan=True)
