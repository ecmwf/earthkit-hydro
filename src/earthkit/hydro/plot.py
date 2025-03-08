import numpy as np


def plot(river_network, field, ax=None):

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "matplotlib.pyplot is required for plotting."
            "\nTo install it, run `pip install matplotlib`"
        )

    x = np.arange(river_network.mask.shape[1])
    y = np.arange(river_network.mask.shape[0])[::-1]

    X, Y = np.meshgrid(x, y)
    x = X[river_network.mask]
    y = Y[river_network.mask]

    not_sinks = river_network.downstream_nodes != river_network.n_nodes
    u = np.zeros_like(x)
    u[not_sinks] = x[river_network.downstream_nodes[not_sinks]] - x[not_sinks]
    v = np.zeros_like(x)
    v[not_sinks] = y[river_network.downstream_nodes[not_sinks]] - y[not_sinks]

    c = field[river_network.mask]

    if ax is None:
        _, ax = plt.subplots()

    ax.quiver(x, y, u, v, c, scale=1, scale_units="xy", angles="xy")
    return ax
