import numpy as np

# catchment_query_field_1 = np.array(
#     [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1, np.nan, 5, 4, 2, 3, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], dtype="int"
# )
catchment_query_field_1 = [8, 12, 13, 11, 10]


# catchment_query_field_2 = np.array(
#     [4, np.nan, 1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 3, np.nan, 2, np.nan, np.nan], dtype="int"
# )
catchment_query_field_2 = [2, 13, 11, 0]


# subcatchment_1 = np.array([5, 4, 2, 2, 1, 5, 4, 2, 1, 3, 5, 4, 2, 3, 3, np.nan, np.nan, np.nan, np.nan, np.nan])-1


# subcatchment_2 = np.array([4, 1, 1, np.nan, 2, 2, 2, 3, 2, 2, 2, 3, 4, 2, 3, 3])-1


catchment_1 = (
    np.array(
        [
            5,
            4,
            2,
            2,
            2,
            5,
            4,
            2,
            2,
            2,
            5,
            4,
            2,
            2,
            2,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ]
    )
    - 1
)


catchment_2 = np.array([4, 2, 2, np.nan, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2]) - 1

# Catchment aggregation results for catchment_query_field_1 with input_field_1c
# Locations: [8, 12, 13, 11, 10]
catchment_sum_1c = np.array([5.6, 23.6, 0.3, 23.0, 9.8])
catchment_mean_1c = np.array([2.8, 2.62222222, 0.1, 7.66666667, 3.26666667])
catchment_min_1c = np.array([0.1, -4.5, -4.5, 4.6, 1.5])
catchment_max_1c = np.array([5.5, 8.9, 3.3, 11.1, 5.1])
