import numpy as np

# RIVER NETWORK ONE

# 1a: unit field input
input_field_1a = np.array(
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int
)

mv_1a = np.iinfo(np.int64).max

upstream_metric_sum_1a = np.array(
    [1, 1, 1, 1, 1, 2, 2, 3, 2, 1, 3, 3, 9, 3, 1, 1, 20, 3, 2, 1], dtype=int
)

upstream_metric_max_1a = np.array(
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int
)

upstream_metric_min_1a = np.array(
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int
)

upstream_metric_mean_1a = np.array(
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=float
)

upstream_metric_product_1a = np.array(
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int
)

# 1b: non-missing integer field input
input_field_1b = np.array(
    [1, 2, 3, -1, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, -1], dtype=int
)

mv_1b = np.iinfo(np.int64).max

upstream_metric_sum_1b = np.array(
    [1, 2, 3, -1, 5, 7, 9, 10, 14, 10, 8, 11, 46, 19, 5, 6, 94, 16, 8, -1], dtype=int
)

upstream_metric_max_1b = np.array(
    [1, 2, 3, -1, 5, 6, 7, 8, 9, 10, 6, 7, 10, 10, 5, 6, 10, 9, 9, -1], dtype=int
)

upstream_metric_min_1b = np.array(
    [1, 2, 3, -1, 5, 1, 2, -1, 5, 10, 1, 2, -1, 4, 5, 6, -1, -1, -1, -1], dtype=int
)

upstream_metric_mean_1b = np.array(
    [
        1.0,
        2.0,
        3.0,
        -1.0,
        5.0,
        3.5,
        4.5,
        3.333333333,
        7.0,
        10.0,
        2.666666667,
        3.666666667,
        5.111111111,
        6.333333333,
        5.0,
        6.0,
        4.7,
        5.333333333,
        4.0,
        -1.0,
    ],
    dtype=float,
)

upstream_metric_product_1b = np.array(
    [
        1,
        2,
        3,
        -1,
        5,
        6,
        14,
        -24,
        45,
        10,
        6,
        28,
        -648000,
        200,
        5,
        6,
        329204736000,
        -72,
        -9,
        -1,
    ],
    dtype=int,
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

upstream_metric_sum_1c = np.array(
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

upstream_metric_max_1c = np.array(
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

upstream_metric_min_1c = np.array(
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

upstream_metric_mean_1c = np.array(
    [
        1.5,
        7.3,
        5.6,
        -3.2,
        0.1,
        3.3,
        9.2,
        3.76666667,
        2.8,
        1.5,
        3.26666667,
        7.66666667,
        2.62222222,
        0.1,
        -4.5,
        -8.9,
        2.805,
        3.56666667,
        2.75,
        1.1,
    ],
    dtype=float,
)

upstream_metric_product_1c = np.array(
    [
        1.50000000e00,
        7.30000000e00,
        5.60000000e00,
        -3.20000000e00,
        1.00000000e-01,
        7.65000000e00,
        8.10300000e01,
        -1.59488000e02,
        5.50000000e-01,
        1.50000000e00,
        2.44800000e01,
        3.72738000e02,
        1.25051351e04,
        -2.22750000e01,
        -4.50000000e00,
        -8.90000000e00,
        5.36736931e10,
        2.51680000e01,
        4.84000000e00,
        1.10000000e00,
    ],
    dtype=float,
)

# 1d: bool field input
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

mv_1d = False

upstream_metric_sum_1d = np.array(
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

upstream_metric_max_1d = upstream_metric_sum_1d

upstream_metric_min_1d = upstream_metric_sum_1d

upstream_metric_mean_1d = upstream_metric_sum_1d.astype("float")

upstream_metric_product_1d = upstream_metric_sum_1d

# 1e: missing float field input with mv=np.nan
input_field_1e = np.array(
    [
        1.5,
        7.3,
        5.6,
        -3.2,
        0.1,
        5.1,
        11.1,
        np.nan,
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
        np.nan,
        1.1,
    ],
    dtype=float,
)

mv_1e = np.nan

upstream_metric_sum_1e = np.array(
    [
        1.5,
        7.3,
        5.6,
        -3.2,
        0.1,
        6.6,
        18.4,
        np.nan,
        5.6,
        1.5,
        9.8,
        23.0,
        np.nan,
        0.3,
        -4.5,
        -8.9,
        np.nan,
        np.nan,
        np.nan,
        1.1,
    ],
    dtype=float,
)

upstream_metric_max_1e = np.array(
    [
        1.5,
        7.3,
        5.6,
        -3.2,
        0.1,
        5.1,
        11.1,
        np.nan,
        5.5,
        1.5,
        5.1,
        11.1,
        np.nan,
        3.3,
        -4.5,
        -8.9,
        np.nan,
        np.nan,
        np.nan,
        1.1,
    ],
    dtype=float,
)

upstream_metric_min_1e = np.array(
    [
        1.5,
        7.3,
        5.6,
        -3.2,
        0.1,
        1.5,
        7.3,
        np.nan,
        0.1,
        1.5,
        1.5,
        4.6,
        np.nan,
        -4.5,
        -4.5,
        -8.9,
        np.nan,
        np.nan,
        np.nan,
        1.1,
    ],
    dtype=float,
)

upstream_metric_mean_1e = np.array(
    [
        1.5,
        7.3,
        5.6,
        -3.2,
        0.1,
        3.3,
        9.2,
        np.nan,
        2.8,
        1.5,
        3.266666666666667,
        7.666666666666667,
        np.nan,
        0.1,
        -4.5,
        -8.9,
        np.nan,
        np.nan,
        np.nan,
        1.1,
    ],
    dtype=float,
)

upstream_metric_product_1e = np.array(
    [
        1.50000e00,
        7.30000e00,
        5.60000e00,
        -3.20000e00,
        1.00000e-01,
        7.65000e00,
        8.10300e01,
        np.nan,
        5.50000e-01,
        1.50000e00,
        2.44800e01,
        3.72738e02,
        np.nan,
        -2.22750e01,
        -4.50000e00,
        -8.90000e00,
        np.nan,
        np.nan,
        np.nan,
        1.10000e00,
    ],
    dtype=float,
)

# 1f: missing float field input with mv=0
input_field_1f = np.nan_to_num(input_field_1e, nan=0)

mv_1f = 0

upstream_metric_sum_1f = np.nan_to_num(upstream_metric_sum_1e, nan=0)

upstream_metric_max_1f = np.nan_to_num(upstream_metric_max_1e, nan=0)

upstream_metric_min_1f = np.nan_to_num(upstream_metric_min_1e, nan=0)

upstream_metric_mean_1f = np.nan_to_num(upstream_metric_mean_1e, nan=0)

upstream_metric_product_1f = np.nan_to_num(upstream_metric_product_1e, nan=0)


# 1g: missing integer field input with mv=-1
input_field_1g = np.array(
    [1, 2, 3, -1, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, -1], dtype=int
)

mv_1g = -1

upstream_metric_sum_1g = np.array(
    [1, 2, 3, -1, 5, 7, 9, -1, 14, 10, 8, 11, -1, 19, 5, 6, -1, -1, -1, -1], dtype=int
)

upstream_metric_max_1g = np.array(
    [1, 2, 3, -1, 5, 6, 7, -1, 9, 10, 6, 7, -1, 10, 5, 6, -1, -1, -1, -1], dtype=int
)

upstream_metric_min_1g = np.array(
    [1, 2, 3, -1, 5, 1, 2, -1, 5, 10, 1, 2, -1, 4, 5, 6, -1, -1, -1, -1], dtype=int
)

upstream_metric_mean_1g = np.array(
    [
        1.0,
        2.0,
        3.0,
        -1.0,
        5.0,
        3.5,
        4.5,
        -1.0,
        7.0,
        10.0,
        2.6666666666666665,
        3.6666666666666665,
        -1.0,
        6.333333,
        5.0,
        6.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
    ],
    dtype=float,
)

upstream_metric_product_1g = np.array(
    [1, 2, 3, -1, 5, 6, 14, -1, 45, 10, 6, 28, -1, 200, 5, 6, -1, -1, -1, -1], dtype=int
)

# RIVER NETWORK TWO

# 2a: unit field input
input_field_2a = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int)

mv_2a = np.iinfo(np.int64).max

upstream_metric_sum_2a = np.array(
    [2, 1, 2, 1, 1, 2, 7, 3, 1, 1, 10, 6, 1, 13, 1, 2], dtype=int
)

upstream_metric_max_2a = np.array(
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int
)

upstream_metric_min_2a = np.array(
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int
)

upstream_metric_mean_2a = np.array(
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=float
)

upstream_metric_product_2a = np.array(
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int
)

# 2b: non-missing integer field input
input_field_2b = np.array(
    [1, 2, -1, 4, 5, 6, 7, 8, 9, 10, 11, 12, -1, 14, 15, 16], dtype=int
)

mv_2b = np.iinfo(np.int64).max

upstream_metric_sum_2b = np.array(
    [0, 2, 1, 4, 5, 11, 59, 9, 9, 10, 81, 52, -1, 114, 15, 31], dtype=int
)

upstream_metric_max_2b = np.array(
    [1, 2, 2, 4, 5, 6, 16, 8, 9, 10, 16, 16, -1, 16, 15, 16], dtype=int
)

upstream_metric_min_2b = np.array(
    [-1, 2, -1, 4, 5, 5, -1, -1, 9, 10, -1, -1, -1, -1, 15, 15], dtype=int
)

upstream_metric_mean_2b = np.array(
    [
        0.0,
        2.0,
        0.5,
        4.0,
        5.0,
        5.5,
        8.428571,
        3.0,
        9.0,
        10.0,
        8.1,
        8.666667,
        -1.0,
        8.769231,
        15.0,
        15.5,
    ],
    dtype=float,
)

upstream_metric_product_2b = np.array(
    [
        -1,
        2,
        -2,
        4,
        5,
        30,
        -322560,
        -16,
        9,
        10,
        -106444800,
        -46080,
        -1,
        -134120448000,
        15,
        240,
    ],
    dtype=int,
)

# 2c: non-missing float field input

# 2d: bool field input

# 2e: missing float field input with mv=np.nan

# 2f: missing float field input with mv=0

# 2g: missing integer field input with mv=-1

input_field_2g = np.array(
    [1, 2, -1, 4, 5, 6, 7, 8, 9, 10, 11, 12, -1, 14, 15, 16], dtype=int
)

mv_2g = -1

upstream_metric_sum_2g = np.array(
    [-1, 2, -1, 4, 5, 11, -1, -1, 9, 10, -1, -1, -1, -1, 15, 31], dtype=int
)
