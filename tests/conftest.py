import pytest

from earthkit.hydro._readers import from_cama_downxy, from_cama_nextxy, from_d8
from earthkit.hydro.data_structures import RiverNetwork


@pytest.fixture
def river_network(request):
    river_network_format, flow_directions = request.param
    if river_network_format == "d8_ldd":
        river_network = from_d8(flow_directions)
    elif river_network_format == "cama_downxy":
        river_network = from_cama_downxy(*flow_directions)
    elif river_network_format == "cama_nextxy":
        river_network = from_cama_nextxy(*flow_directions)
    # TODO: add ESRI

    return RiverNetwork(river_network)
