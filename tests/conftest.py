import pytest
import numpy as np


@pytest.fixture
def d8_ldd():
    return np.array(
        [
            [2, 2, 2, 1, 1],
            [2, 2, 2, 1, 1],
            [3, 2, 1, 4, 4],
            [6, 5, 4, 4, 4],
        ]
    )


@pytest.fixture
def d4_ldd():
    return np.array(
        [
            [3, 3, 3, 4, 3],
            [3, 3, 3, 4, 4],
            [2, 3, 3, 4, 4],
            [2, 0, 2, 2, 2],
        ]
    )


@pytest.fixture
def cama_drain():
    return (
        np.array(
            [
                [1, 2, 3, 3, 4],
                [1, 2, 3, 3, 4],
                [2, 2, 2, 3, 4],
                [2, -10, 2, 3, 4],
            ]
        ),
        np.array(
            [
                [2, 2, 2, 2, 2],
                [3, 3, 3, 3, 3],
                [4, 4, 4, 3, 3],
                [4, -10, 4, 4, 4],
            ]
        ),
    )


@pytest.fixture
def downstream_nodes():
    return np.array(
        [
            5,
            6,
            7,
            7,
            8,
            10,
            11,
            12,
            12,
            13,
            16,
            16,
            16,
            12,
            13,
            16,
            20,  # we set sink to len of nodes
            16,
            17,
            18,
        ]
    )


@pytest.fixture
def upstream_points():
    return np.array([1, 1, 1, 1, 1, 2, 2, 3, 2, 1, 3, 3, 9, 3, 1, 1, 20, 3, 2, 1])


@pytest.fixture
def test_field():
    return np.arange(1, 21, dtype=int)


@pytest.fixture
def upstream():
    return np.array([0, 0, 0, 0, 0, 1, 2, 7, 5, 0, 6, 7, 31, 25, 0, 0, 70, 19, 20, 0])


@pytest.fixture
def downstream():
    return np.array([6, 7, 8, 8, 9, 11, 12, 13, 13, 14, 17, 17, 17, 13, 14, 17, 0, 17, 18, 19])
