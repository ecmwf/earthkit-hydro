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
            upstream_metric_mean_1c,
            mv_1c,
        ),
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1e,
            upstream_metric_mean_1e,
            mv_1e,
        ),
    ],
    indirect=["river_network"],
)
def test_calculate_upstream_metric_mean(
    river_network, input_field, flow_downstream, mv
):
    output_field = ekh.upstream.array.mean(
        river_network, input_field, node_weights=None, return_type="masked"
    )
    assert output_field.dtype == flow_downstream.dtype
    np.testing.assert_allclose(output_field, flow_downstream)

    input_field = convert_to_2d(river_network, input_field, 0)
    flow_downstream = convert_to_2d(river_network, flow_downstream, 0)
    output_field = ekh.upstream.array.mean(
        river_network,
        input_field,
        node_weights=None,
    ).flatten()
    print(output_field)
    print(flow_downstream)
    assert output_field.dtype == flow_downstream.dtype
    np.testing.assert_allclose(output_field, flow_downstream)
