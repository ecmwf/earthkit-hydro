from earthkit.hydro._utils.decorators import xarray
from earthkit.hydro.downstream import array


@xarray
def var(
    river_network,
    field,
    node_weights=None,
    edge_weights=None,
    return_type=None,
    input_core_dims=None,
):
    r"""
    Computes the weighted variance of a field over all downstream nodes.

    For each node in the river network, this function identifies all downstream nodes in the river network
    and accumulates their contributions upstream, weighted by both node and edge weights.

    The weighted variance is defined as:

    .. math::
       :nowrap:

       \begin{align*}
       x'_i &= w'_i \cdot x_i \\
       q'_i &= w'_i \cdot x_i^2 \\
       n_j &= x'_j + \sum_{i \in \mathrm{Down}(j)} w_{ij} \cdot n_i \\
       q_j &= q'_j + \sum_{i \in \mathrm{Down}(j)} w_{ij} \cdot q_i \\
       d_j &= w'_j + \sum_{i \in \mathrm{Down}(j)} w_{ij} \cdot d_i \\
       \bar{x}_j &= \frac{n_j}{d_j} \\
       \mathrm{Var}(x)_j &= \frac{q_j}{d_j} - \bar{x}_j^2
       \end{align*}

    where:

    - :math:`x_i` is the input value at node :math:`i` (e.g., rainfall),
    - :math:`w'_i` is the node weight (e.g., pixel area),
    - :math:`w_{ij}` is the edge weight from node :math:`i` to node :math:`j` (e.g., discharge partitioning ratio),
    - :math:`\mathrm{Down}(j)` is the set of downstream nodes flowing out of node :math:`j`,
    - :math:`n_j` is the accumulated weighted value,
    - :math:`q_j` is the accumulated weighted squared value,
    - :math:`d_j` is the accumulated weight (denominator),
    - :math:`\bar{x}_j` is the weighted average at node :math:`j`,
    - :math:`\mathrm{Var}(x)_j` is the weighted variance at node :math:`j`.

    Accumulation proceeds in inverse topological order from the sinks to the sources. This formulation computes the population variance.

    Parameters
    ----------
    river_network : RiverNetwork
        A river network object.
    field : array-like or xarray object
        An array containing field values defined on river network nodes or gridcells.
    node_weights : array-like or xarray object, optional
        Array of weights for each river network node or gridcell. Default is None (unweighted).
    edge_weights : array-like or xarray object, optional
        Array of weights for each edge. Default is None (unweighted).
    return_type : str, optional
        Either "masked", "gridded" or None. If None (default), uses `river_network.return_type`.
    input_core_dims : sequence of sequence, optional
        List of core dimensions on each input xarray argument that should not be broadcast.
        Default is None, which attempts to autodetect input_core_dims from the xarray inputs.
        Ignored if no xarray inputs passed.

    Returns
    -------
    xarray object
        Array of variance values for every river network node or gridcell, depending on `return_type`.
    """
    return array.var(river_network, field, node_weights, edge_weights, return_type)


@xarray
def std(
    river_network,
    field,
    node_weights=None,
    edge_weights=None,
    return_type=None,
    input_core_dims=None,
):
    r"""
    Computes the weighted standard deviation of a field over all
    downstream nodes.

    For each node in the river network, this function identifies all downstream nodes in the river network
    and accumulates their contributions upstream, weighted by both node and edge weights.

    The weighted standard deviation is defined as:

    .. math::
       :nowrap:

       \begin{align*}
       x'_i &= w'_i \cdot x_i \\
       q'_i &= w'_i \cdot x_i^2 \\
       n_j &= x'_j + \sum_{i \in \mathrm{Down}(j)} w_{ij} \cdot n_i \\
       q_j &= q'_j + \sum_{i \in \mathrm{Down}(j)} w_{ij} \cdot q_i \\
       d_j &= w'_j + \sum_{i \in \mathrm{Down}(j)} w_{ij} \cdot d_i \\
       \bar{x}_j &= \frac{n_j}{d_j} \\
       \mathrm{Var}(x)_j &= \frac{q_j}{d_j} - \bar{x}_j^2 \\
       \mathrm{Std}(x)_j &= \sqrt{\mathrm{Var}(x)_j}
       \end{align*}

    where:

    - :math:`x_i` is the input value at node :math:`i` (e.g., rainfall),
    - :math:`w'_i` is the node weight (e.g., pixel area),
    - :math:`w_{ij}` is the edge weight from node :math:`i` to node :math:`j` (e.g., discharge partitioning ratio),
    - :math:`\mathrm{Down}(j)` is the set of downstream nodes flowing out of node :math:`j`,
    - :math:`n_j` is the accumulated weighted value,
    - :math:`q_j` is the accumulated weighted squared value,
    - :math:`d_j` is the accumulated weight (denominator),
    - :math:`\bar{x}_j` is the weighted average at node :math:`j`,
    - :math:`\mathrm{Var}(x)_j` is the weighted variance at node :math:`j`.
    - :math:`\mathrm{Std}(x)_j` is the weighted standard deviation at node :math:`j`.

    Accumulation proceeds in inverse topological order from the sinks to the sources. This formulation computes the population standard deviation.

    Parameters
    ----------
    river_network : RiverNetwork
        A river network object.
    field : array-like or xarray object
        An array containing field values defined on river network nodes or gridcells.
    node_weights : array-like or xarray object, optional
        Array of weights for each river network node or gridcell. Default is None (unweighted).
    edge_weights : array-like or xarray object, optional
        Array of weights for each edge. Default is None (unweighted).
    return_type : str, optional
        Either "masked", "gridded" or None. If None (default), uses `river_network.return_type`.
    input_core_dims : sequence of sequence, optional
        List of core dimensions on each input xarray argument that should not be broadcast.
        Default is None, which attempts to autodetect input_core_dims from the xarray inputs.
        Ignored if no xarray inputs passed.

    Returns
    -------
    xarray object
        Array of standard deviation values for every river network node or gridcell, depending on `return_type`.
    """
    return array.std(river_network, field, node_weights, edge_weights, return_type)


@xarray
def mean(
    river_network,
    field,
    node_weights=None,
    edge_weights=None,
    return_type=None,
    input_core_dims=None,
):
    r"""
    Computes the weighted mean of a field over all downstream nodes.

    For each node in the river network, this function identifies all downstream nodes in the river network
    and accumulates their contributions upstream, weighted by both node and edge weights.

    The weighted mean is defined as:

    .. math::
       :nowrap:

       \begin{align*}
       x'_i &= w'_i \cdot x_i \\
       n_j &= x'_j + \sum_{i \in \mathrm{Down}(j)} w_{ij} \cdot n_i \\
       d_j &= w'_j + \sum_{i \in \mathrm{Down}(j)} w_{ij} \cdot d_i \\
       \bar{x}_j &= \frac{n_j}{d_j}
       \end{align*}

    where:

    - :math:`x_i` is the input value at node :math:`i` (e.g., rainfall),
    - :math:`w'_i` is the node weight (e.g., pixel area),
    - :math:`w_{ij}` is the edge weight from node :math:`i` to node :math:`j` (e.g. discharge partitioning ratio),
    - :math:`\mathrm{Down}(j)` is the set of downstream nodes flowing out of node :math:`j`,
    - :math:`n_j` is the accumulated weighted value,
    - :math:`d_j` is the accumulated weight (denominator),
    - :math:`\bar{x}_j` is the weighted mean at node :math:`j`.

    Accumulation proceeds in inverse topological order from the sinks to the sources.

    Parameters
    ----------
    river_network : RiverNetwork
        A river network object.
    field : array-like or xarray object
        An array containing field values defined on river network nodes or gridcells.
    node_weights : array-like or xarray object, optional
        Array of weights for each river network node or gridcell. Default is None (unweighted).
    edge_weights : array-like or xarray object, optional
        Array of weights for each edge. Default is None (unweighted).
    return_type : str, optional
        Either "masked", "gridded" or None. If None (default), uses `river_network.return_type`.
    input_core_dims : sequence of sequence, optional
        List of core dimensions on each input xarray argument that should not be broadcast.
        Default is None, which attempts to autodetect input_core_dims from the xarray inputs.
        Ignored if no xarray inputs passed.

    Returns
    -------
    xarray object
        Array of mean values for every river network node or gridcell, depending on `return_type`.
    """
    return array.mean(river_network, field, node_weights, edge_weights, return_type)


@xarray
def sum(
    river_network,
    field,
    node_weights=None,
    edge_weights=None,
    return_type=None,
    input_core_dims=None,
):
    r"""
    Computes the weighted sum of a field over all downstream nodes.

    For each node in the river network, this function identifies all downstream nodes in the river network
    and accumulates their contributions upstream, weighted by both node and edge weights.

    The weighted sum is defined as:

    .. math::
        :nowrap:

        \begin{align*}
        x'_i &= w'_i \cdot x_i \\
        n_j &= x'_j + \sum_{i \in \mathrm{Down}(j)} w_{ij} \cdot n_i
        \end{align*}

    where:

    - :math:`x_i` is the input value at node :math:`i` (e.g., rainfall),
    - :math:`w'_i` is the node weight (e.g., pixel area),
    - :math:`w_{ij}` is the edge weight from node :math:`i` to node :math:`j` (e.g. discharge partitioning ratio),
    - :math:`\mathrm{Down}(j)` is the set of downstream nodes flowing out of node :math:`j`,
    - :math:`n_j` is the weighted sum at node :math:`j`.

    Accumulation proceeds in inverse topological order from the sinks to the sources.

    Parameters
    ----------
    river_network : RiverNetwork
        A river network object.
    field : array-like or xarray object
        An array containing field values defined on river network nodes or gridcells.
    node_weights : array-like or xarray object, optional
        Array of weights for each river network node or gridcell. Default is None (unweighted).
    edge_weights : array-like or xarray object, optional
        Array of weights for each edge. Default is None (unweighted).
    return_type : str, optional
        Either "masked", "gridded" or None. If None (default), uses `river_network.return_type`.
    input_core_dims : sequence of sequence, optional
        List of core dimensions on each input xarray argument that should not be broadcast.
        Default is None, which attempts to autodetect input_core_dims from the xarray inputs.
        Ignored if no xarray inputs passed.

    Returns
    -------
    xarray object
        Array of sum values for every river network node or gridcell, depending on `return_type`.
    """
    return array.sum(river_network, field, node_weights, edge_weights, return_type)


@xarray
def min(
    river_network,
    field,
    node_weights=None,
    edge_weights=None,
    return_type=None,
    input_core_dims=None,
):
    r"""
    Computes the weighted minimum of a field over all downstream nodes.

    For each node in the river network, this function identifies all downstream nodes in the river network
    and accumulates their contributions upstream, weighted by both node and edge weights.

    The weighted minimum is defined as:

    .. math::
        :nowrap:

        \begin{align*}
        x'_i &= w'_i \cdot x_i \\
        m_j &= \mathrm{min}(x'_j,~\mathrm{min}_{i \in \mathrm{Down}(j)} w_{ij} \cdot m_i)
        \end{align*}

    where:

    - :math:`x_i` is the input value at node :math:`i` (e.g., rainfall),
    - :math:`w'_i` is the node weight (e.g., pixel area),
    - :math:`w_{ij}` is the edge weight from node :math:`i` to node :math:`j` (e.g. discharge partitioning ratio),
    - :math:`\mathrm{Down}(j)` is the set of downstream nodes flowing out of node :math:`j`,
    - :math:`m_j` is the weighted minimum at node :math:`j`.

    Accumulation proceeds in inverse topological order from the sinks to the sources.

    Parameters
    ----------
    river_network : RiverNetwork
        A river network object.
    field : array-like or xarray object
        An array containing field values defined on river network nodes or gridcells.
    node_weights : array-like or xarray object, optional
        Array of weights for each river network node or gridcell. Default is None (unweighted).
    edge_weights : array-like or xarray object, optional
        Array of weights for each edge. Default is None (unweighted).
    return_type : str, optional
        Either "masked", "gridded" or None. If None (default), uses `river_network.return_type`.
    input_core_dims : sequence of sequence, optional
        List of core dimensions on each input xarray argument that should not be broadcast.
        Default is None, which attempts to autodetect input_core_dims from the xarray inputs.
        Ignored if no xarray inputs passed.

    Returns
    -------
    xarray object
        Array of minimum values for every river network node or gridcell, depending on `return_type`.
    """
    return array.min(river_network, field, node_weights, edge_weights, return_type)


@xarray
def max(
    river_network,
    field,
    node_weights=None,
    edge_weights=None,
    return_type=None,
    input_core_dims=None,
):
    r"""
    Computes the weighted maximum of a field over all downstream nodes.

    For each node in the river network, this function identifies all downstream nodes in the river network
    and accumulates their contributions upstream, weighted by both node and edge weights.

    The weighted maximum is defined as:

    .. math::
        :nowrap:

        \begin{align*}
        x'_i &= w'_i \cdot x_i \\
        m_j &= \mathrm{max} (x'_j,~\mathrm{max}_{i \in \mathrm{Down}(j)} w_{ij} \cdot m_i)
        \end{align*}

    where:

    - :math:`x_i` is the input value at node :math:`i` (e.g., rainfall),
    - :math:`w'_i` is the node weight (e.g., pixel area),
    - :math:`w_{ij}` is the edge weight from node :math:`i` to node :math:`j` (e.g. discharge partitioning ratio),
    - :math:`\mathrm{Down}(j)` is the set of downstream nodes flowing out of node :math:`j`,
    - :math:`m_j` is the weighted maximum at node :math:`j`.

    Accumulation proceeds in inverse topological order from the sinks to the sources.

    Parameters
    ----------
    river_network : RiverNetwork
        A river network object.
    field : array-like or xarray object
        An array containing field values defined on river network nodes or gridcells.
    node_weights : array-like or xarray object, optional
        Array of weights for each river network node or gridcell. Default is None (unweighted).
    edge_weights : array-like or xarray object, optional
        Array of weights for each edge. Default is None (unweighted).
    return_type : str, optional
        Either "masked", "gridded" or None. If None (default), uses `river_network.return_type`.
    input_core_dims : sequence of sequence, optional
        List of core dimensions on each input xarray argument that should not be broadcast.
        Default is None, which attempts to autodetect input_core_dims from the xarray inputs.
        Ignored if no xarray inputs passed.

    Returns
    -------
    xarray object
        Array of maximum values for every river network node or gridcell, depending on `return_type`.
    """
    return array.max(river_network, field, node_weights, edge_weights, return_type)
