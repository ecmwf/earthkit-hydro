import earthkit.hydro.catchments.array.__operations as array
from earthkit.hydro._utils.decorators import multi_backend


@multi_backend(allow_jax_jit=False)
def var(
    xp,
    river_network,
    field,
    locations,
    node_weights,
    edge_weights,
):
    return array.var(xp, river_network, field, locations, node_weights, edge_weights)


@multi_backend(allow_jax_jit=False)
def std(
    xp,
    river_network,
    field,
    locations,
    node_weights,
    edge_weights,
):
    return array.std(xp, river_network, field, locations, node_weights, edge_weights)


@multi_backend(allow_jax_jit=False)
def mean(
    xp,
    river_network,
    field,
    locations,
    node_weights,
    edge_weights,
):
    return array.mean(xp, river_network, field, locations, node_weights, edge_weights)


@multi_backend(allow_jax_jit=False)
def sum(
    xp,
    river_network,
    field,
    locations,
    node_weights,
    edge_weights,
):
    return array.sum(xp, river_network, field, locations, node_weights, edge_weights)


@multi_backend(allow_jax_jit=False)
def min(
    xp,
    river_network,
    field,
    locations,
    node_weights,
    edge_weights,
):
    return array.min(xp, river_network, field, locations, node_weights, edge_weights)


@multi_backend(allow_jax_jit=False)
def max(
    xp,
    river_network,
    field,
    locations,
    node_weights,
    edge_weights,
):
    return array.max(xp, river_network, field, locations, node_weights, edge_weights)
