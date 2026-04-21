import numpy as np
import pytest

from earthkit.hydro._readers import from_cama_downxy, from_cama_nextxy, from_d8
from earthkit.hydro.data_structures import RiverNetwork


@pytest.fixture
def river_network(request):
    river_network_format, flow_directions = request.param
    if river_network_format == "d8_ldd":
        river_network_storage = from_d8(flow_directions)
    elif river_network_format == "cama_downxy":
        river_network_storage = from_cama_downxy(*flow_directions)
    elif river_network_format == "cama_nextxy":
        river_network_storage = from_cama_nextxy(*flow_directions)
    # TODO: add ESRI

    # Add coords manually since we're using internal _readers
    # (when using river_network.create with earthkit-data, coords are auto-extracted)
    if river_network_storage.coords is None:
        ny, nx = river_network_storage.shape
        river_network_storage.coords = {"y": np.arange(ny), "x": np.arange(nx)}

    return RiverNetwork(river_network_storage)
