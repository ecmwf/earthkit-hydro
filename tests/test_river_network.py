import pytest
from test_inputs.readers import *
from test_inputs.subnetwork import *

import earthkit.hydro as ekh


@pytest.mark.parametrize(
    "river_network, mask, accumulate_downstream",
    [
        (("d8_ldd", d8_ldd_2), mask_2, masked_unit_accuflux_2),
        (("cama_downxy", cama_downxy_2), mask_2, masked_unit_accuflux_2),
        (("cama_nextxy", cama_nextxy_2), mask_2, masked_unit_accuflux_2),
    ],
    indirect=["river_network"],
)
def test_subnetwork(river_network, mask, accumulate_downstream):
    river_network = river_network.create_subnetwork(mask)
    field = np.ones(river_network.n_nodes)
    accum = ekh.flow_downstream(river_network, field)
    print(accum)
    print(accumulate_downstream)
    np.testing.assert_array_equal(accum, accumulate_downstream)
