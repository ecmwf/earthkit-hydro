import numpy as np

# RIVER NETWORK ONE

# 1a: unit field input
input_field_1a = np.array(
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int
)

mv_1a = np.nan

flow_downstream_sum_1a = np.array(
    [1, 1, 1, 1, 1, 2, 2, 3, 2, 1, 3, 3, 9, 3, 1, 1, 20, 3, 2, 1], dtype=int
)

flow_downstream_max_1a = np.array(
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int
)

flow_downstream_min_1a = np.array(
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int
)

# 1b: non-missing integer field input
input_field_1b = np.array(
    [1, 2, 3, -1, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, -1], dtype=int
)

mv_1b = np.nan

flow_downstream_sum_1b = np.array(
    [1, 2, 3, -1, 5, 7, 9, 10, 14, 10, 8, 11, 46, 19, 5, 6, 94, 16, 8, -1], dtype=int
)

flow_downstream_max_1b = np.array(
    [1, 2, 3, -1, 5, 6, 7, 8, 9, 10, 6, 7, 10, 10, 5, 6, 10, 9, 9, -1], dtype=int
)

flow_downstream_min_1b = np.array(
    [1, 2, 3, -1, 5, 1, 2, -1, 5, 10, 1, 2, -1, 4, 5, 6, -1, -1, -1, -1], dtype=int
)

# 1c: non-missing float field input
input_field_1c = np.array(
    [
        1.5,
        7.3,
        5.6,
        -3.2,
        0.1,
        5.1,
        11.1,
        8.9,
        5.5,
        1.5,
        3.2,
        4.6,
        6.4,
        3.3,
        -4.5,
        -8.9,
        -2.1,
        5.2,
        4.4,
        1.1,
    ],
    dtype=float,
)

mv_1c = np.nan

flow_downstream_sum_1c = np.array(
    [
        1.5,
        7.3,
        5.6,
        -3.2,
        0.1,
        6.6,
        18.4,
        11.3,
        5.6,
        1.5,
        9.8,
        23.0,
        23.6,
        0.3,
        -4.5,
        -8.9,
        56.1,
        10.7,
        5.5,
        1.1,
    ],
    dtype=float,
)

flow_downstream_max_1c = np.array(
    [
        1.5,
        7.3,
        5.6,
        -3.2,
        0.1,
        5.1,
        11.1,
        8.9,
        5.5,
        1.5,
        5.1,
        11.1,
        8.9,
        3.3,
        -4.5,
        -8.9,
        11.1,
        5.2,
        4.4,
        1.1,
    ],
    dtype=float,
)

flow_downstream_min_1c = np.array(
    [
        1.5,
        7.3,
        5.6,
        -3.2,
        0.1,
        1.5,
        7.3,
        -3.2,
        0.1,
        1.5,
        1.5,
        4.6,
        -4.5,
        -4.5,
        -4.5,
        -8.9,
        -8.9,
        1.1,
        1.1,
        1.1,
    ],
    dtype=float,
)

# 1d: non-missing bool field input
input_field_1d = np.array(
    [
        True,
        False,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        False,
        False,
        True,
        True,
        True,
        False,
        False,
        True,
        False,
    ],
    dtype=bool,
)

mv_1d = np.nan

flow_downstream_sum_1d = np.array(
    [
        True,
        False,
        True,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        True,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
    ],
    dtype=bool,
)

flow_downstream_max_1d = np.array(
    [
        True,
        False,
        True,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        True,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
    ],
    dtype=bool,
)

flow_downstream_min_1d = np.array(
    [
        True,
        False,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
    ],
    dtype=bool,
)

# 1e: missing float field input with mv=np.nan

# 1f: missing float field input with mv=0

# 1g: missing integer field input with mv=-1

input_field_1g = np.array(
    [1, 2, 3, -1, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, -1], dtype=int
)

mv_1g = -1

flow_downstream_sum_1g = np.array(
    [1, 2, 3, -1, 5, 7, 9, -1, 14, 10, 8, 11, -1, 19, 5, 6, -1, -1, -1, -1], dtype=int
)

flow_downstream_max_1g = np.array(
    [1, 2, 3, -1, 5, 6, 7, -1, 9, 10, 6, 7, -1, 10, 5, 6, -1, -1, -1, -1], dtype=int
)

flow_downstream_min_1g = np.array(
    [1, 2, 3, -1, 5, 1, 2, -1, 5, 10, 1, 2, -1, 4, 5, 6, -1, -1, -1, -1], dtype=int
)

# 1h: missing bool field input with mv=False


# RIVER NETWORK TWO

# 2a: unit field input
input_field_2a = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int)

mv_2a = np.nan

flow_downstream_sum_2a = np.array(
    [2, 1, 2, 1, 1, 2, 7, 3, 1, 1, 10, 6, 1, 13, 1, 2], dtype=int
)

# 2b: non-missing integer field input
input_field_2b = np.array(
    [1, 2, -1, 4, 5, 6, 7, 8, 9, 10, 11, 12, -1, 14, 15, 16], dtype=int
)

mv_2b = np.nan

flow_downstream_sum_2b = np.array(
    [0, 2, 1, 4, 5, 11, 59, 9, 9, 10, 81, 52, -1, 114, 15, 31], dtype=int
)

# 2c: non-missing float field input

# 2d: non-missing bool field input

# 2e: missing float field input with mv=np.nan

# 2f: missing float field input with mv=0

# 2g: missing integer field input with mv=-1

input_field_2g = np.array(
    [1, 2, -1, 4, 5, 6, 7, 8, 9, 10, 11, 12, -1, 14, 15, 16], dtype=int
)

mv_2g = -1

flow_downstream_sum_2g = np.array(
    [-1, 2, -1, 4, 5, 11, -1, -1, 9, 10, -1, -1, -1, -1, 15, 31], dtype=int
)
