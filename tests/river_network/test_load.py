# SPDX-FileCopyrightText: 2026- European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

import earthkit.hydro as ekh


def test_load():
    net = ekh.river_network.load("efas", "5", use_cache=False)
    assert net.n_nodes == 7446075
    assert net.n_edges == 7353055
