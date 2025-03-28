import numpy as np
import pytest
from test_inputs.catchment import *
from test_inputs.readers import *

import earthkit.hydro as ekh


@pytest.mark.parametrize(
    "river_network",
    [
        ("d8_ldd", d8_ldd_1),
        ("cama_downxy", cama_downxy_1),
        ("cama_nextxy", cama_nextxy_1),
        ("d8_ldd", d8_ldd_2),
        ("cama_downxy", cama_downxy_2),
        ("cama_nextxy", cama_nextxy_2),
    ],
    indirect=True,
)
def test_find_subcatchments_does_not_overwrite(river_network):
    field = np.arange(river_network.n_nodes) + 1
    subcatchments = ekh.find_subcatchments(river_network, field)
    print(subcatchments)
    print(field)
    np.testing.assert_array_equal(subcatchments, field)


@pytest.mark.parametrize(
    "river_network",
    [
        ("d8_ldd", d8_ldd_1),
        ("cama_downxy", cama_downxy_1),
        ("cama_nextxy", cama_nextxy_1),
        ("d8_ldd", d8_ldd_2),
        ("cama_downxy", cama_downxy_2),
        ("cama_nextxy", cama_nextxy_2),
    ],
    indirect=True,
)
def test_find_subcatchments_does_not_overwrite_2d(river_network):
    field = np.zeros(river_network.mask.shape, dtype="int")
    field[river_network.mask] = np.arange(river_network.n_nodes) + 1
    find_subcatchments = ekh.find_subcatchments(river_network, field)
    print(find_subcatchments)
    print(field)
    np.testing.assert_array_equal(find_subcatchments, field)


@pytest.mark.parametrize(
    "river_network, query_field, subcatchment",
    [
        (("d8_ldd", d8_ldd_1), catchment_query_field_1, subcatchment_1),
        (("cama_downxy", cama_downxy_1), catchment_query_field_1, subcatchment_1),
        (("cama_nextxy", cama_nextxy_1), catchment_query_field_1, subcatchment_1),
        (("d8_ldd", d8_ldd_2), catchment_query_field_2, subcatchment_2),
        (("cama_downxy", cama_downxy_2), catchment_query_field_2, subcatchment_2),
        (("cama_nextxy", cama_nextxy_2), catchment_query_field_2, subcatchment_2),
    ],
    indirect=["river_network"],
)
def test_find_subcatchments(river_network, query_field, subcatchment):
    subcatchments = ekh.find_subcatchments(river_network, query_field)
    print(subcatchment)
    print(subcatchments)
    np.testing.assert_array_equal(subcatchment, subcatchments)


@pytest.mark.parametrize(
    "river_network, query_field, find_subcatchments",
    [
        (("d8_ldd", d8_ldd_1), catchment_query_field_1, subcatchment_1),
        (("cama_downxy", cama_downxy_1), catchment_query_field_1, subcatchment_1),
        (("cama_nextxy", cama_nextxy_1), catchment_query_field_1, subcatchment_1),
        (("d8_ldd", d8_ldd_2), catchment_query_field_2, subcatchment_2),
        (("cama_downxy", cama_downxy_2), catchment_query_field_2, subcatchment_2),
        (("cama_nextxy", cama_nextxy_2), catchment_query_field_2, subcatchment_2),
    ],
    indirect=["river_network"],
)
def test_find_subcatchments_2d(river_network, query_field, find_subcatchments):
    field = np.zeros(river_network.mask.shape, dtype="int")
    field[river_network.mask] = query_field
    network_find_subcatchments = ekh.find_subcatchments(river_network, field)
    print(find_subcatchments)
    print(network_find_subcatchments)
    np.testing.assert_array_equal(
        network_find_subcatchments[river_network.mask], find_subcatchments
    )
    np.testing.assert_array_equal(network_find_subcatchments[~river_network.mask], 0)


@pytest.mark.parametrize(
    "river_network, query_field, find_subcatchments",
    [
        (("d8_ldd", d8_ldd_1), catchment_query_field_1, subcatchment_1),
        (("cama_downxy", cama_downxy_1), catchment_query_field_1, subcatchment_1),
        (("cama_nextxy", cama_nextxy_1), catchment_query_field_1, subcatchment_1),
        (("d8_ldd", d8_ldd_2), catchment_query_field_2, subcatchment_2),
        (("cama_downxy", cama_downxy_2), catchment_query_field_2, subcatchment_2),
        (("cama_nextxy", cama_nextxy_2), catchment_query_field_2, subcatchment_2),
    ],
    indirect=["river_network"],
)
def test_find_subcatchments_Nd(river_network, query_field, find_subcatchments):
    field = np.zeros(river_network.mask.shape, dtype="int")
    field[river_network.mask] = query_field
    field = np.stack([field, field], axis=0)
    network_find_subcatchments = ekh.find_subcatchments(river_network, field)
    find_subcatchments = np.stack([find_subcatchments, find_subcatchments], axis=0)
    print(find_subcatchments)
    print(network_find_subcatchments)
    np.testing.assert_array_equal(
        network_find_subcatchments[..., river_network.mask], find_subcatchments
    )
    np.testing.assert_array_equal(
        network_find_subcatchments[..., ~river_network.mask], 0
    )


@pytest.mark.parametrize(
    "river_network, query_field, find_catchments",
    [
        (("d8_ldd", d8_ldd_1), catchment_query_field_1, catchment_1),
        (("cama_downxy", cama_downxy_1), catchment_query_field_1, catchment_1),
        (("cama_nextxy", cama_nextxy_1), catchment_query_field_1, catchment_1),
        (("d8_ldd", d8_ldd_2), catchment_query_field_2, catchment_2),
        (("cama_downxy", cama_downxy_2), catchment_query_field_2, catchment_2),
        (("cama_nextxy", cama_nextxy_2), catchment_query_field_2, catchment_2),
    ],
    indirect=["river_network"],
)
def test_find_catchments_2d(river_network, query_field, find_catchments):
    field = np.zeros(river_network.mask.shape, dtype="int")
    field[river_network.mask] = query_field
    network_find_catchments = ekh.find_catchments(river_network, field)
    print(find_catchments)
    print(network_find_catchments)
    np.testing.assert_array_equal(
        network_find_catchments[river_network.mask], find_catchments
    )
    np.testing.assert_array_equal(network_find_catchments[~river_network.mask], 0)


@pytest.mark.parametrize(
    "river_network, query_field, find_catchments",
    [
        (("d8_ldd", d8_ldd_1), catchment_query_field_1, catchment_1),
        (("cama_downxy", cama_downxy_1), catchment_query_field_1, catchment_1),
        (("cama_nextxy", cama_nextxy_1), catchment_query_field_1, catchment_1),
        (("d8_ldd", d8_ldd_2), catchment_query_field_2, catchment_2),
        (("cama_downxy", cama_downxy_2), catchment_query_field_2, catchment_2),
        (("cama_nextxy", cama_nextxy_2), catchment_query_field_2, catchment_2),
    ],
    indirect=["river_network"],
)
def test_find_catchments_Nd(river_network, query_field, find_catchments):
    field = np.zeros(river_network.mask.shape, dtype="int")
    field[river_network.mask] = query_field
    field = np.stack([field, field], axis=0)
    network_find_catchments = ekh.find_catchments(river_network, field)
    find_catchments = np.stack([find_catchments, find_catchments], axis=0)
    print(find_catchments)
    print(network_find_catchments)
    np.testing.assert_array_equal(
        network_find_catchments[..., river_network.mask], find_catchments
    )
    np.testing.assert_array_equal(network_find_catchments[..., ~river_network.mask], 0)
