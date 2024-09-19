import pytest


@pytest.fixture
def d8_ldd():
    return [
        [2, 2, 2, 1, 1],
        [2, 2, 2, 1, 1],
        [3, 2, 1, 4, 4],
        [6, 5, 4, 4, 4],
    ]


@pytest.fixture
def d4_ldd():
    return [
        [3, 3, 3, 4, 3],
        [3, 3, 3, 4, 4],
        [2, 3, 3, 4, 4],
        [2, 0, 2, 2, 2],
    ]


@pytest.fixture
def cama_drain():
    return (
        [
            [1, 2, 3, 3, 4],
            [1, 2, 3, 3, 4],
            [2, 2, 2, 3, 4],
            [2, -10, 2, 3, 4],
        ],
        [
            [2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3],
            [4, 4, 4, 3, 3],
            [4, -10, 4, 4, 4],
        ],
    )


@pytest.fixture
def downstream_list():
    return [
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
        -1,
        16,
        17,
        18,
    ]

@pytest.fixture
def unit_field_accuflux_list():
    return [1,1,1,1,1,2,2,3,2,1,3,3,9,3,1,1,20,3,2,1]