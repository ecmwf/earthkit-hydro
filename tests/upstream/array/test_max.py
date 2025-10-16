import numpy as np
import pytest
from _test_inputs.accumulation import *
from _test_inputs.readers import *

import earthkit.hydro as ekh


@pytest.mark.parametrize(
    "river_network, input_field, flow_downstream, mv",
    [
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1c,
            upstream_metric_max_1c,
            mv_1c,
        ),
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1e,
            upstream_metric_max_1e,
            mv_1e,
        ),
    ],
    indirect=["river_network"],
)
def test_calculate_upstream_metric_max(river_network, input_field, flow_downstream, mv):
    output_field = ekh.upstream.array.max(
        river_network, input_field, node_weights=None, return_type="masked"
    )
    print(output_field)
    print(flow_downstream)
    assert output_field.dtype == flow_downstream.dtype
    np.testing.assert_allclose(output_field, flow_downstream)
