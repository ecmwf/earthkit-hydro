import earthkit.hydro.streamorder.array as array
from earthkit.hydro._utils.decorators import xarray


@xarray
def strahler(
    river_network,
    return_type=None,
    input_core_dims=None,
):
    r"""
    Computes the Strahler stream order for each node in the river network.

    For each node in the river network, this algorithm identifies the upstream nodes
    and uses them to calculate the Strahler number of the current node.

    The Strahler number is defined as:

    .. math::
       :nowrap:

       \begin{align*}
       u_i &= \max_{k \in \mathrm{Up}(i)} s_k \\
       n_i &= \left|\{k \in \mathrm{Up}(i) | s_k = u_i\}\right| \\
       s_i &= \begin{cases}
       1 & \text{if } n_i = 0 \\
       u_i & \text{if } n_i = 1 \\
       u_i + 1 & \text{if } n_i \geq 2
       \end{cases}
       \end{align*}

    where:

    - :math:`u_i` is the maximum Strahler number among the upstream nodes of node :math:`i`,
    - :math:`\mathrm{Up}(j)` is the set of upstream nodes flowing into node :math:`j`,
    - :math:`n_i` is the count of upstream nodes with the maximum Strahler number $u_i$,
    - :math:`s_i` is the Strahler number at node :math:`i`.

    Accumulation proceeds in topological order from the sources to the sinks.

    Parameters
    ----------
    river_network : RiverNetwork
        A river network object.
    return_type : str, optional
        Either "masked", "gridded" or None. If None (default), uses `river_network.return_type`.
    input_core_dims : sequence of sequence, optional
        List of core dimensions on each input xarray argument that should not be broadcast.
        Default is None, which attempts to autodetect input_core_dims from the xarray inputs.
        Ignored if no xarray inputs passed.

    Returns
    -------
    xarray object
        Array of Strahler stream order values for every river network node or gridcell, depending on `return_type`.
    """
    return array.strahler(river_network=river_network, return_type=return_type)


@xarray
def shreve(
    river_network,
    return_type=None,
    input_core_dims=None,
):
    r"""
    Computes the Shreve stream order for each node in the river network.

    For each node in the river network, this algorithm identifies the upstream nodes
    and uses them to calculate the Shreve order of the current node.

    The Shreve order is defined as:

    .. math::
       :nowrap:

       \begin{align*}
       s_i &= 1 ~\text{for sources}\\
       s_i &= \sum_{k \in \mathrm{Up}(i)} s_k
       \end{align*}

    where:

    - :math:`\mathrm{Up}(j)` is the set of upstream nodes flowing into node :math:`j`,
    - :math:`s_i` is the Shreve order at node :math:`i`.

    Accumulation proceeds in topological order from the sources to the sinks.

    Parameters
    ----------
    river_network : RiverNetwork
        A river network object.
    return_type : str, optional
        Either "masked", "gridded" or None. If None (default), uses `river_network.return_type`.
    input_core_dims : sequence of sequence, optional
        List of core dimensions on each input xarray argument that should not be broadcast.
        Default is None, which attempts to autodetect input_core_dims from the xarray inputs.
        Ignored if no xarray inputs passed.

    Returns
    -------
    xarray object
        Array of Shreve stream order values for every river network node or gridcell, depending on `return_type`.
    """
    return array.shreve(river_network=river_network, return_type=return_type)
