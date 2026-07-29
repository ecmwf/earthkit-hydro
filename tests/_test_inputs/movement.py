# SPDX-FileCopyrightText: 2026- European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

import numpy as np

upstream_1 = np.array([0, 0, 0, 0, 0, 1, 2, 7, 5, 0, 6, 7, 31, 25, 0, 0, 70, 19, 20, 0], dtype=float)


upstream_2 = np.array([13, 0, 2, 0, 0, 5, 12, 3, 0, 0, 13, 24, 0, 30, 0, 15], dtype=float)


downstream_1 = np.array(
    [6, 7, 8, 8, 9, 11, 12, 13, 13, 14, 17, 17, 17, 13, 14, 17, 0, 17, 18, 19],
    dtype=float,
)


downstream_2 = np.array([0, 3, 8, 0, 6, 11, 11, 12, 14, 14, 14, 7, 1, 0, 16, 12], dtype=float)

# Move upstream skewness for network 1
move_upstream_skewness_1 = np.array(
    [
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ],
    dtype=float,
)

# Move downstream skewness for network 1
move_downstream_skewness_1 = np.array(
    [
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        0.0,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        0.6309038567106233,
        0.0,
        np.nan,
        np.nan,
        0.40604028821330246,
        np.nan,
        np.nan,
        np.nan,
    ],
    dtype=float,
)

# Move upstream skewness for network 2
move_upstream_skewness_2 = np.array(
    [
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ],
    dtype=float,
)

# Move downstream skewness for network 2
move_downstream_skewness_2 = np.array(
    [
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        0.0,
        0.0,
        np.nan,
        0.0,
        np.nan,
        np.nan,
    ],
    dtype=float,
)
