import pytest

import earthkit.hydro as ekh


@pytest.fixture
def river_network(request):
    river_network_format, flow_directions = request.param
    if river_network_format == "d8_ldd":
        river_network = ekh.readers.from_d8(flow_directions)
    elif river_network_format == "cama_downxy":
        river_network = ekh.readers.from_cama_downxy(*flow_directions)
    elif river_network_format == "cama_nextxy":
        river_network = ekh.readers.from_cama_nextxy(*flow_directions)
    # TODO: add ESRI

    return river_network
