from conftest import *
from helper import read_network
from pytest_cases import parametrize

import earthkit.hydro as ekh


@parametrize(
    "reader,map_name,mask,accumulate_downstream",
    [
        ("d8_ldd", d8_ldd_2, mask_2, masked_unit_accuflux_2),
        ("cama_downxy", cama_downxy_2, mask_2, masked_unit_accuflux_2),
        ("cama_nextxy", cama_nextxy_2, mask_2, masked_unit_accuflux_2),
    ],
)
def test_subnetwork(reader, map_name, mask, accumulate_downstream):
    network = read_network(reader, map_name)
    network = network.create_subnetwork(mask)
    field = np.ones(network.n_nodes)
    accum = ekh.flow_downstream(network, field)
    print(accum)
    print(accumulate_downstream)
    np.testing.assert_array_equal(accum, accumulate_downstream)
