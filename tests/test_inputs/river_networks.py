import numpy as np

downstream_nodes_1 = np.array(
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


downstream_nodes_2 = np.array(
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
