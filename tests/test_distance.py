# @parametrize(
#     "reader,map_name,distance_query,distance",
#     [
#         ("d8_ldd", d8_ldd_1, distance_query_field_1, distance_1),
#         ("cama_downxy", cama_downxy_1, distance_query_field_1, distance_1),
#         ("cama_nextxy", cama_nextxy_1, distance_query_field_1, distance_1),
#     ],
# )
# def test_distance(reader, map_name, distance_query, distance):
#     network = read_network(reader, map_name)
#     dist = ekh.compute_distance(network, distance_query)
#     print(dist)
#     print(distance)
#     np.testing.assert_array_equal(dist, distance)


# @parametrize(
#     "reader,map_name,distance_query,distance",
#     [
#         ("d8_ldd", d8_ldd_1, distance_query_field_1, distance_1),
#         ("cama_downxy", cama_downxy_1, distance_query_field_1, distance_1),
#         ("cama_nextxy", cama_nextxy_1, distance_query_field_1, distance_1),
#     ],
# )
# def test_distance_2d(reader, map_name, distance_query, distance):
#     network = read_network(reader, map_name)
#     field = np.zeros(network.mask.shape, dtype="int") - 1
#     field[network.mask] = distance_query
#     network_dist = ekh.compute_distance(network, field)
#     print(distance)
#     print(network_dist)
#     np.testing.assert_array_equal(network_dist[network.mask], distance)
#     np.testing.assert_array_equal(network_dist[~network.mask], -1)


# @parametrize(
#     "reader,map_name,distance_query,distance",
#     [
#         ("d8_ldd", d8_ldd_1, distance_query_field_1, distance_1),
#         ("cama_downxy", cama_downxy_1, distance_query_field_1, distance_1),
#         ("cama_nextxy", cama_nextxy_1, distance_query_field_1, distance_1),
#     ],
# )
# def test_distance_Nd(reader, map_name, distance_query, distance):
#     network = read_network(reader, map_name)
#     field = np.zeros(network.mask.shape, dtype="int") - 1
#     field[network.mask] = distance_query
#     field = np.stack([field, field], axis=0)
#     network_dist = ekh.compute_distance(network, field)
#     distance = np.stack([distance, distance], axis=0)
#     print(distance)
#     print(network_dist)
#     np.testing.assert_array_equal(network_dist[..., network.mask], distance)
#     np.testing.assert_array_equal(network_dist[..., ~network.mask], -1)
