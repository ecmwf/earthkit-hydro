import numpy as np
from pytest_cases import fixture


@fixture
def d8_ldd_1():
    return np.array(
        [
            [2, 2, 2, 1, 1],
            [2, 2, 2, 1, 1],
            [3, 2, 1, 4, 4],
            [6, 5, 4, 4, 4],
        ]
    )


@fixture
def d8_ldd_2():
    return np.array(
        [
            [5, 6, 3, 5],
            [6, 3, 2, 2],
            [3, 2, 1, 7],
            [2, 5, 6, 8],
        ]
    )


@fixture
def cama_downxy_1():
    return (
        np.array(
            [
                [0, 0, 0, -1, -1],
                [0, 0, 0, -1, -1],
                [1, 0, -1, -1, -1],
                [1, -999, -1, -1, -1],
            ]
        ),
        np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 0, 0],
                [0, -999, 0, 0, 0],
            ]
        ),
    )


@fixture
def cama_downxy_2():
    return (
        np.array(
            [
                [-999, 1, 1, -999],
                [1, 1, 0, 0],
                [1, 0, -1, -1],
                [0, -999, 1, 0],
            ]
        ),
        np.array(
            [
                [-999, 0, 1, -999],
                [0, 1, 1, 1],
                [1, 1, 1, -1],
                [1, -999, 0, -1],
            ]
        ),
    )


@fixture
def cama_nextxy_1():
    return (
        np.array(
            [
                [1, 2, 3, 3, 4],
                [1, 2, 3, 3, 4],
                [2, 2, 2, 3, 4],
                [2, -9, 2, 3, 4],
            ]
        ),
        np.array(
            [
                [2, 2, 2, 2, 2],
                [3, 3, 3, 3, 3],
                [4, 4, 4, 3, 3],
                [4, -9, 4, 4, 4],
            ]
        ),
    )


@fixture
def cama_nextxy_2():
    return (
        np.array(
            [
                [-9, 3, 4, -9],
                [2, 3, 3, 4],
                [2, 2, 2, 3],
                [1, -9, 4, 4],
            ]
        ),
        np.array(
            [
                [-9, 1, 2, -9],
                [2, 3, 3, 3],
                [4, 4, 4, 2],
                [1, -9, 4, 3],
            ]
        ),
    )


@fixture
def downstream_nodes_1():
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


@fixture
def downstream_nodes_2():
    return np.array(
        [
            16,  # we set sink to len of nodes
            2,
            7,
            16,  # we set sink to len of nodes
            5,
            10,
            10,
            11,
            13,
            13,
            13,
            6,
            0,
            16,  # we set sink to len of nodes
            15,
            11,
        ]
    )


@fixture
def unit_field_accuflux_1():
    return np.array([1, 1, 1, 1, 1, 2, 2, 3, 2, 1, 3, 3, 9, 3, 1, 1, 20, 3, 2, 1])


@fixture
def unit_field_accuflux_2():
    return np.array([2, 1, 2, 1, 1, 2, 7, 3, 1, 1, 10, 6, 1, 13, 1, 2])


@fixture
def upstream_1():
    return np.array([0, 0, 0, 0, 0, 1, 2, 7, 5, 0, 6, 7, 31, 25, 0, 0, 70, 19, 20, 0])


@fixture
def upstream_2():
    return np.array([13, 0, 2, 0, 0, 5, 12, 3, 0, 0, 13, 24, 0, 30, 0, 15])


@fixture
def downstream_1():
    return np.array([6, 7, 8, 8, 9, 11, 12, 13, 13, 14, 17, 17, 17, 13, 14, 17, 0, 17, 18, 19])


@fixture
def downstream_2():
    return np.array([0, 3, 8, 0, 6, 11, 11, 12, 14, 14, 14, 7, 1, 0, 16, 12])
