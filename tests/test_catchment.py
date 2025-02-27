import numpy as np
from conftest import *
from helper import read_network
from pytest_cases import parametrize

import earthkit.hydro as ekh


@parametrize(
    "reader,map_name",
    [
        ("d8_ldd", d8_ldd_1),
        ("cama_downxy", cama_downxy_1),
        ("cama_nextxy", cama_nextxy_1),
        ("d8_ldd", d8_ldd_2),
        ("cama_downxy", cama_downxy_2),
        ("cama_nextxy", cama_nextxy_2),
    ],
)
def test_find_subcatchments_does_not_overwrite(reader, map_name):
    network = read_network(reader, map_name)
    field = np.arange(network.n_nodes) + 1
    subcatchments = ekh.find_subcatchments(network, field)
    print(subcatchments)
    print(field)
    np.testing.assert_array_equal(subcatchments, field)


@parametrize(
    "reader,map_name",
    [
        ("d8_ldd", d8_ldd_1),
        ("cama_downxy", cama_downxy_1),
        ("cama_nextxy", cama_nextxy_1),
        ("d8_ldd", d8_ldd_2),
        ("cama_downxy", cama_downxy_2),
        ("cama_nextxy", cama_nextxy_2),
    ],
)
def test_find_subcatchments_does_not_overwrite_2d(reader, map_name):
    network = read_network(reader, map_name)
    field = np.zeros(network.mask.shape, dtype="int")
    field[network.mask] = np.arange(network.n_nodes) + 1
    find_subcatchments = ekh.find_subcatchments(network, field)
    print(find_subcatchments)
    print(field)
    np.testing.assert_array_equal(find_subcatchments, field)


@parametrize(
    "reader,map_name,query_field,subcatchment",
    [
        ("d8_ldd", d8_ldd_1, catchment_query_field_1, subcatchment_1),
        ("cama_downxy", cama_downxy_1, catchment_query_field_1, subcatchment_1),
        ("cama_nextxy", cama_nextxy_1, catchment_query_field_1, subcatchment_1),
        ("d8_ldd", d8_ldd_2, catchment_query_field_2, subcatchment_2),
        ("cama_downxy", cama_downxy_2, catchment_query_field_2, subcatchment_2),
        ("cama_nextxy", cama_nextxy_2, catchment_query_field_2, subcatchment_2),
    ],
)
def test_find_subcatchments(reader, map_name, query_field, subcatchment):
    network = read_network(reader, map_name)
    subcatchments = ekh.find_subcatchments(network, query_field)
    print(subcatchment)
    print(subcatchments)
    np.testing.assert_array_equal(subcatchment, subcatchments)


@parametrize(
    "reader,map_name,query_field,find_subcatchments",
    [
        ("d8_ldd", d8_ldd_1, catchment_query_field_1, subcatchment_1),
        ("cama_downxy", cama_downxy_1, catchment_query_field_1, subcatchment_1),
        ("cama_nextxy", cama_nextxy_1, catchment_query_field_1, subcatchment_1),
        ("d8_ldd", d8_ldd_2, catchment_query_field_2, subcatchment_2),
        ("cama_downxy", cama_downxy_2, catchment_query_field_2, subcatchment_2),
        ("cama_nextxy", cama_nextxy_2, catchment_query_field_2, subcatchment_2),
    ],
)
def test_find_subcatchments_2d(reader, map_name, query_field, find_subcatchments):
    network = read_network(reader, map_name)
    field = np.zeros(network.mask.shape, dtype="int")
    field[network.mask] = query_field
    network_find_subcatchments = ekh.find_subcatchments(network, field)
    print(find_subcatchments)
    print(network_find_subcatchments)
    np.testing.assert_array_equal(
        network_find_subcatchments[network.mask], find_subcatchments
    )
    np.testing.assert_array_equal(network_find_subcatchments[~network.mask], 0)


@parametrize(
    "reader,map_name,query_field,find_subcatchments",
    [
        ("d8_ldd", d8_ldd_1, catchment_query_field_1, subcatchment_1),
        ("cama_downxy", cama_downxy_1, catchment_query_field_1, subcatchment_1),
        ("cama_nextxy", cama_nextxy_1, catchment_query_field_1, subcatchment_1),
        ("d8_ldd", d8_ldd_2, catchment_query_field_2, subcatchment_2),
        ("cama_downxy", cama_downxy_2, catchment_query_field_2, subcatchment_2),
        ("cama_nextxy", cama_nextxy_2, catchment_query_field_2, subcatchment_2),
    ],
)
def test_find_subcatchments_Nd(reader, map_name, query_field, find_subcatchments):
    network = read_network(reader, map_name)
    field = np.zeros(network.mask.shape, dtype="int")
    field[network.mask] = query_field
    field = np.stack([field, field], axis=0)
    network_find_subcatchments = ekh.find_subcatchments(network, field)
    find_subcatchments = np.stack([find_subcatchments, find_subcatchments], axis=0)
    print(find_subcatchments)
    print(network_find_subcatchments)
    np.testing.assert_array_equal(
        network_find_subcatchments[..., network.mask], find_subcatchments
    )
    np.testing.assert_array_equal(network_find_subcatchments[..., ~network.mask], 0)


@parametrize(
    "reader,map_name,query_field,find_catchments",
    [
        ("d8_ldd", d8_ldd_1, catchment_query_field_1, catchment_1),
        ("cama_downxy", cama_downxy_1, catchment_query_field_1, catchment_1),
        ("cama_nextxy", cama_nextxy_1, catchment_query_field_1, catchment_1),
        ("d8_ldd", d8_ldd_2, catchment_query_field_2, catchment_2),
        ("cama_downxy", cama_downxy_2, catchment_query_field_2, catchment_2),
        ("cama_nextxy", cama_nextxy_2, catchment_query_field_2, catchment_2),
    ],
)
def test_find_catchments_2d(reader, map_name, query_field, find_catchments):
    network = read_network(reader, map_name)
    field = np.zeros(network.mask.shape, dtype="int")
    field[network.mask] = query_field
    network_find_catchments = ekh.find_catchments(network, field)
    print(find_catchments)
    print(network_find_catchments)
    np.testing.assert_array_equal(
        network_find_catchments[network.mask], find_catchments
    )
    np.testing.assert_array_equal(network_find_catchments[~network.mask], 0)


@parametrize(
    "reader,map_name,query_field,find_catchments",
    [
        ("d8_ldd", d8_ldd_1, catchment_query_field_1, catchment_1),
        ("cama_downxy", cama_downxy_1, catchment_query_field_1, catchment_1),
        ("cama_nextxy", cama_nextxy_1, catchment_query_field_1, catchment_1),
        ("d8_ldd", d8_ldd_2, catchment_query_field_2, catchment_2),
        ("cama_downxy", cama_downxy_2, catchment_query_field_2, catchment_2),
        ("cama_nextxy", cama_nextxy_2, catchment_query_field_2, catchment_2),
    ],
)
def test_find_catchments_Nd(reader, map_name, query_field, find_catchments):
    network = read_network(reader, map_name)
    field = np.zeros(network.mask.shape, dtype="int")
    field[network.mask] = query_field
    field = np.stack([field, field], axis=0)
    network_find_catchments = ekh.find_catchments(network, field)
    find_catchments = np.stack([find_catchments, find_catchments], axis=0)
    print(find_catchments)
    print(network_find_catchments)
    np.testing.assert_array_equal(
        network_find_catchments[..., network.mask], find_catchments
    )
    np.testing.assert_array_equal(network_find_catchments[..., ~network.mask], 0)
