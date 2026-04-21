from earthkit.hydro._core.online import calculate_online_metric
from earthkit.hydro._utils.decorators import mask, multi_backend


def calculate_downstream_metric(
    xp,
    river_network,
    field,
    metric,
    node_weights,
    edge_weights,
):
    return calculate_online_metric(
        xp,
        river_network,
        field,
        metric,
        node_weights,
        edge_weights,
        flow_direction="up",
    )


# TODO: clean up
def percentile(river_network, field, weights, p, return_type):
    from earthkit.hydro import _rust

    def calculate_percentile(xp, river_network, field, weights, p):
        if weights is not None:
            return _rust.calc_weighted_perc_downstream(
                river_network.groups, field, weights, p
            )
        else:
            return _rust.calc_perc_downstream(river_network.groups, field, p)

    return_type = river_network.return_type if return_type is None else return_type
    if return_type not in ["gridded", "masked"]:
        raise ValueError("return_type must be either 'gridded' or 'masked'.")
    decorated_calculate_downstream_metric = mask(return_type == "gridded")(
        calculate_percentile
    )
    # TODO: assert inputs are numpy
    from earthkit.hydro._backends.numpy_backend import NumPyBackend

    return decorated_calculate_downstream_metric(
        NumPyBackend(), river_network, field, weights, p  # ignored
    )


@multi_backend(jax_static_args=["xp", "river_network", "return_type"])
def var(xp, river_network, field, node_weights, edge_weights, return_type):
    return_type = river_network.return_type if return_type is None else return_type
    if return_type not in ["gridded", "masked"]:
        raise ValueError("return_type must be either 'gridded' or 'masked'.")
    decorated_calculate_downstream_metric = mask(return_type == "gridded")(
        calculate_downstream_metric
    )
    return decorated_calculate_downstream_metric(
        xp,
        river_network,
        field,
        "var",
        node_weights,
        edge_weights,
    )


@multi_backend(jax_static_args=["xp", "river_network", "return_type"])
def std(xp, river_network, field, node_weights, edge_weights, return_type):
    return_type = river_network.return_type if return_type is None else return_type
    if return_type not in ["gridded", "masked"]:
        raise ValueError("return_type must be either 'gridded' or 'masked'.")
    decorated_calculate_downstream_metric = mask(return_type == "gridded")(
        calculate_downstream_metric
    )
    return decorated_calculate_downstream_metric(
        xp,
        river_network,
        field,
        "std",
        node_weights,
        edge_weights,
    )


@multi_backend(jax_static_args=["xp", "river_network", "return_type"])
def mean(xp, river_network, field, node_weights, edge_weights, return_type):
    return_type = river_network.return_type if return_type is None else return_type
    if return_type not in ["gridded", "masked"]:
        raise ValueError("return_type must be either 'gridded' or 'masked'.")
    decorated_calculate_downstream_metric = mask(return_type == "gridded")(
        calculate_downstream_metric
    )
    return decorated_calculate_downstream_metric(
        xp,
        river_network,
        field,
        "mean",
        node_weights,
        edge_weights,
    )


@multi_backend(jax_static_args=["xp", "river_network", "return_type"])
def sum(xp, river_network, field, node_weights, edge_weights, return_type):
    return_type = river_network.return_type if return_type is None else return_type
    if return_type not in ["gridded", "masked"]:
        raise ValueError("return_type must be either 'gridded' or 'masked'.")
    decorated_calculate_downstream_metric = mask(return_type == "gridded")(
        calculate_downstream_metric
    )
    return decorated_calculate_downstream_metric(
        xp,
        river_network,
        field,
        "sum",
        node_weights,
        edge_weights,
    )


@multi_backend(jax_static_args=["xp", "river_network", "return_type"])
def min(xp, river_network, field, node_weights, edge_weights, return_type):
    return_type = river_network.return_type if return_type is None else return_type
    if return_type not in ["gridded", "masked"]:
        raise ValueError("return_type must be either 'gridded' or 'masked'.")
    decorated_calculate_downstream_metric = mask(return_type == "gridded")(
        calculate_downstream_metric
    )
    return decorated_calculate_downstream_metric(
        xp,
        river_network,
        field,
        "min",
        node_weights,
        edge_weights,
    )


@multi_backend(jax_static_args=["xp", "river_network", "return_type"])
def max(xp, river_network, field, node_weights, edge_weights, return_type):
    return_type = river_network.return_type if return_type is None else return_type
    if return_type not in ["gridded", "masked"]:
        raise ValueError("return_type must be either 'gridded' or 'masked'.")
    decorated_calculate_downstream_metric = mask(return_type == "gridded")(
        calculate_downstream_metric
    )
    return decorated_calculate_downstream_metric(
        xp,
        river_network,
        field,
        "max",
        node_weights,
        edge_weights,
    )


@multi_backend(jax_static_args=["xp", "river_network", "return_type"])
def mode(xp, river_network, field, node_weights, edge_weights, return_type):
    from earthkit.hydro import _rust

    def calculate_mode(xp, river_network, field, node_weights, edge_weights):
        # Mode only supported for numpy backend with Rust
        if xp.name != "numpy":
            raise NotImplementedError(
                "Mode is only supported for numpy backend with Rust"
            )
        return _rust.calc_mode_downstream(river_network.groups, field)

    return_type = river_network.return_type if return_type is None else return_type
    if return_type not in ["gridded", "masked"]:
        raise ValueError("return_type must be either 'gridded' or 'masked'.")
    decorated_calculate_mode = mask(return_type == "gridded")(calculate_mode)
    # TODO: assert inputs are numpy
    from earthkit.hydro._backends.numpy_backend import NumPyBackend

    return decorated_calculate_mode(
        NumPyBackend(), river_network, field, node_weights, edge_weights  # ignored
    )
