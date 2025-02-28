import numpy as np

catchment_query_field_1 = np.array(
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 5, 4, 2, 3, 0, 0, 0, 0, 0, 0], dtype="int"
)


catchment_query_field_2 = np.array(
    [4, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 2, 0, 0], dtype="int"
)


subcatchment_1 = np.array([5, 4, 2, 2, 1, 5, 4, 2, 1, 3, 5, 4, 2, 3, 3, 0, 0, 0, 0, 0])


subcatchment_2 = np.array([4, 1, 1, 0, 2, 2, 2, 3, 2, 2, 2, 3, 4, 2, 3, 3])


catchment_1 = np.array([5, 4, 2, 2, 2, 5, 4, 2, 2, 2, 5, 4, 2, 2, 2, 0, 0, 0, 0, 0])


catchment_2 = np.array([4, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2])
