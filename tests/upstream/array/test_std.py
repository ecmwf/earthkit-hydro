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
            upstream_metric_std_1c,
            mv_1c,
        ),
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1e,
            upstream_metric_std_1e,
            mv_1e,
        ),
    ],
    indirect=["river_network"],
)
@pytest.mark.parametrize("array_backend", ["numpy", "torch", "jax"])
def test_calculate_upstream_metric_std(river_network, input_field, flow_downstream, mv, array_backend):
    river_network = river_network.to_device("cpu", array_backend)
    xp = ekh._backends.find.get_array_backend(array_backend)
    output_field = ekh.upstream.array.std(
        river_network, xp.asarray(input_field), node_weights=None, return_type="masked"
    )
    output_field = np.asarray(output_field)
    flow_downstream_out = np.asarray(xp.asarray(flow_downstream))
    print(output_field)
    print(flow_downstream_out)
    assert output_field.dtype == flow_downstream_out.dtype
    np.testing.assert_allclose(output_field, flow_downstream, rtol=1e-5, equal_nan=True)
