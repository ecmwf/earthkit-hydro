import pytest

import earthkit.hydro as ekh


@pytest.mark.skip(reason="Environment issue: network not available in test environment")
def test_load():
    net = ekh.river_network.load("efas", "5", use_cache=False)
    assert net.n_nodes == 7446075
    assert net.n_edges == 7353055
