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
            downstream_metric_min_1c,
            mv_1c,
        ),
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1e,
            downstream_metric_min_1e,
            mv_1e,
        ),
    ],
    indirect=["river_network"],
)
def test_downstream_metric_min(river_network, input_field, flow_downstream, mv):
    output_field = ekh.downstream.array.min(
        river_network, input_field, node_weights=None, return_type="masked"
    )
    print(output_field)
    print(flow_downstream)
    assert output_field.dtype == flow_downstream.dtype
    np.testing.assert_allclose(output_field, flow_downstream, equal_nan=True)
