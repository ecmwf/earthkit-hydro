# SPDX-FileCopyrightText: 2026- European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

import numpy as np

# RIVER NETWORK ONE

strahler_1 = np.array(
    [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        2.0,
        1.0,
        1.0,
        1.0,
        1.0,
        3.0,
        2.0,
        1.0,
        1.0,
        3.0,
        1.0,
        1.0,
        1,
    ],
    dtype=int,
)
shreve_1 = np.array(
    [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        2.0,
        1.0,
        1.0,
        1.0,
        1.0,
        5.0,
        2.0,
        1.0,
        1.0,
        9.0,
        1.0,
        1.0,
        1.0,
    ],
    dtype=int,
)

# RIVER NETWORK TWO

strahler_2 = np.array(
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 3.0, 1.0, 1.0],
    dtype=int,
)
shreve_2 = np.array(
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 3.0, 2.0, 1.0, 5.0, 1.0, 1.0],
    dtype=int,
)
