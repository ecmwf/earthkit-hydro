import numpy as np

stations = [(0, 0), (1, 1), (3, 3)]

weights_1 = np.array([6, 1, 2, 3, 4, 7, 1, 5, 5, 0, 6, 1, 0, 9, 9, 8, 3, 0, 6, 4])

distance_1_min_up_down = np.array(
    [
        [0.0, 1.0, 9.0, 10.0, 11.0],
        [6.0, 0.0, 7.0, 7.0, 11.0],
        [8.0, 1.0, 2.0, 11.0, 20.0],
        [10.0, 2.0, 2.0, 0.0, 4.0],
    ]
)

distance_1_min_up = np.array(
    [
        [0.0, 1.0, np.inf, np.inf, np.inf],
        [np.inf, 0.0, np.inf, np.inf, np.inf],
        [np.inf, np.inf, np.inf, np.inf, np.inf],
        [np.inf, np.inf, np.inf, 0.0, 4.0],
    ]
)

distance_1_min_down = np.array(
    [
        [0.0, np.inf, np.inf, np.inf, np.inf],
        [6.0, 0.0, np.inf, np.inf, np.inf],
        [13.0, 1.0, np.inf, np.inf, np.inf],
        [np.inf, 2.0, 6.0, 0.0, np.inf],
    ]
)

length_1_min_up_down = np.array(
    [
        [6.0, 2.0, 12.0, 13.0, 14.0],
        [13.0, 1.0, 10.0, 10.0, 14.0],
        [11.0, 2.0, 5.0, 14.0, 23.0],
        [13.0, 5.0, 5.0, 6.0, 10.0],
    ]
)

length_1_min_up = np.array(
    [
        [6.0, 2.0, np.inf, np.inf, np.inf],
        [np.inf, 1.0, np.inf, np.inf, np.inf],
        [np.inf, np.inf, np.inf, np.inf, np.inf],
        [np.inf, np.inf, np.inf, 6.0, 10.0],
    ]
)

length_1_min_down = np.array(
    [
        [6.0, np.inf, np.inf, np.inf, np.inf],
        [13.0, 1.0, np.inf, np.inf, np.inf],
        [19.0, 2.0, np.inf, np.inf, np.inf],
        [np.inf, 5.0, 6.0, 6.0, np.inf],
    ]
)

# def distance_1_max_up_down():
#     return

distance_1_max_up = np.array(
    [
        [0.0, 1.0, np.inf, np.inf, np.inf],
        [np.inf, 0.0, np.inf, np.inf, np.inf],
        [np.inf, np.inf, np.inf, np.inf, np.inf],
        [np.inf, np.inf, np.inf, 0.0, 4.0],
    ]
)

distance_1_max_down = np.array(
    [
        [0.0, np.inf, np.inf, np.inf, np.inf],
        [6.0, 0.0, np.inf, np.inf, np.inf],
        [13.0, 1.0, np.inf, np.inf, np.inf],
        [np.inf, 19.0, 6.0, 0.0, np.inf],
    ]
)

# def length_1_max_up_down():
#     return

length_1_max_up = np.array(
    [
        [6.0, 2.0, np.inf, np.inf, np.inf],
        [np.inf, 1.0, np.inf, np.inf, np.inf],
        [np.inf, np.inf, np.inf, np.inf, np.inf],
        [np.inf, np.inf, np.inf, 6.0, 10.0],
    ]
)


length_1_max_down = np.array(
    [
        [6.0, np.inf, np.inf, np.inf, np.inf],
        [13.0, 1.0, np.inf, np.inf, np.inf],
        [19.0, 2.0, np.inf, np.inf, np.inf],
        [np.inf, 22.0, 6.0, 6.0, np.inf],
    ]
)
