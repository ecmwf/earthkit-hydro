import numpy as np
import pytest
from _test_inputs.movement import *
from _test_inputs.readers import *

import earthkit.hydro as ekh


@pytest.mark.parametrize(
    "river_network, flow_downstream",
    [
        (("cama_nextxy", cama_nextxy_1), upstream_1),
        (("cama_nextxy", cama_nextxy_2), upstream_2),
    ],
    indirect=["river_network"],
)
def test_calculate_upstream_metric_max(river_network, flow_downstream):
    output_field = ekh.move.array.downstream(
        river_network,
        np.arange(1, river_network.n_nodes + 1),
        node_weights=None,
        return_type="masked",
    )
    print(output_field)
    print(flow_downstream)
    assert output_field.dtype == flow_downstream.dtype
    np.testing.assert_allclose(output_field, flow_downstream)
