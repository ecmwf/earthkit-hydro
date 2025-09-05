import numpy as np
import pytest
from test_inputs.accumulation import *
from test_inputs.readers import *
from utils import convert_to_2d

import earthkit.hydro as ekh


def get_numpy_function(function_name):
    return getattr(np, function_name)


@pytest.mark.parametrize(
    "river_network, input_field, flow_downstream, mv",
    [
        # (
        #     ("cama_nextxy", cama_nextxy_1),
        #     input_field_1a,
        #     upstream_metric_sum_1a,
        #     mv_1a,
        # ),
        # (
        #     ("cama_nextxy", cama_nextxy_1),
        #     input_field_1b,
        #     upstream_metric_sum_1b,
        #     mv_1b,
        # ),
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1c,
            upstream_metric_sum_1c,
            mv_1c,
        ),
        # (
        #     ("cama_nextxy", cama_nextxy_1),
        #     input_field_1d,
        #     upstream_metric_sum_1d,
        #     mv_1d,
        # ),
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1e,
            upstream_metric_sum_1e,
            mv_1e,
        ),
        # (
        #     ("cama_nextxy", cama_nextxy_1),
        #     input_field_1f,
        #     upstream_metric_sum_1f,
        #     mv_1f,
        # ),
        # (
        #     ("cama_nextxy", cama_nextxy_1),
        #     input_field_1g,
        #     upstream_metric_sum_1g,
        #     mv_1g,
        # ),
        # (
        #     ("cama_nextxy", cama_nextxy_2),
        #     input_field_2a,
        #     upstream_metric_sum_2a,
        #     mv_2a,
        # ),
        # (
        #     ("cama_nextxy", cama_nextxy_2),
        #     input_field_2b,
        #     upstream_metric_sum_2b,
        #     mv_2b,
        # ),
        # (
        #     ("cama_nextxy", cama_nextxy_2),
        #     input_field_2g,
        #     upstream_metric_sum_2g,
        #     mv_2g,
        # ),
    ],
    indirect=["river_network"],
)
def test_upstream_metric_sum(river_network, input_field, flow_downstream, mv):
    output_field = ekh.upstream.array.sum(
        river_network, input_field, node_weights=None, return_type="masked"
    )
    print(output_field)
    print(flow_downstream)
    assert output_field.dtype == flow_downstream.dtype
    np.testing.assert_allclose(output_field, flow_downstream)

    print(input_field)
    input_field = convert_to_2d(river_network, input_field, 0)
    flow_downstream = convert_to_2d(river_network, flow_downstream, 0)
    print(mv, input_field.dtype)
    print(input_field, flow_downstream)
    output_field = ekh.upstream.array.sum(
        river_network, input_field, node_weights=None
    ).flatten()
    print(output_field)
    print(flow_downstream)
    assert output_field.dtype == flow_downstream.dtype
    np.testing.assert_allclose(output_field, flow_downstream)


@pytest.mark.parametrize(
    "river_network, input_field, flow_downstream, mv",
    [
        # (
        #     ("cama_nextxy", cama_nextxy_1),
        #     input_field_1a,
        #     upstream_metric_max_1a,
        #     mv_1a,
        # ),
        # (
        #     ("cama_nextxy", cama_nextxy_1),
        #     input_field_1b,
        #     upstream_metric_max_1b,
        #     mv_1b,
        # ),
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1c,
            upstream_metric_max_1c,
            mv_1c,
        ),
        # (
        #     ("cama_nextxy", cama_nextxy_1),
        #     input_field_1d,
        #     upstream_metric_max_1d,
        #     mv_1d,
        # ),
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1e,
            upstream_metric_max_1e,
            mv_1e,
        ),
        # (
        #     ("cama_nextxy", cama_nextxy_1),
        #     input_field_1f,
        #     upstream_metric_max_1f,
        #     mv_1f,
        # ),
        # (
        #     ("cama_nextxy", cama_nextxy_1),
        #     input_field_1g,
        #     upstream_metric_max_1g,
        #     mv_1g,
        # ),
        # (
        #     ("cama_nextxy", cama_nextxy_2),
        #     input_field_2a,
        #     upstream_metric_max_2a,
        #     mv_2a,
        # ),
        # (
        #     ("cama_nextxy", cama_nextxy_2),
        #     input_field_2b,
        #     upstream_metric_max_2b,
        #     mv_2b,
        # ),
    ],
    indirect=["river_network"],
)
def test_calculate_upstream_metric_max(river_network, input_field, flow_downstream, mv):
    output_field = ekh.upstream.array.max(
        river_network, input_field, node_weights=None, return_type="masked"
    )
    print(output_field)
    print(flow_downstream)
    assert output_field.dtype == flow_downstream.dtype
    np.testing.assert_allclose(output_field, flow_downstream)


@pytest.mark.parametrize(
    "river_network, input_field, flow_downstream, mv",
    [
        # (
        #     ("cama_nextxy", cama_nextxy_1),
        #     input_field_1a,
        #     upstream_metric_min_1a,
        #     mv_1a,
        # ),
        # (
        #     ("cama_nextxy", cama_nextxy_1),
        #     input_field_1b,
        #     upstream_metric_min_1b,
        #     mv_1b,
        # ),
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1c,
            upstream_metric_min_1c,
            mv_1c,
        ),
        # (
        #     ("cama_nextxy", cama_nextxy_1),
        #     input_field_1d,
        #     upstream_metric_min_1d,
        #     mv_1d,
        # ),
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1e,
            upstream_metric_min_1e,
            mv_1e,
        ),
        # (
        #     ("cama_nextxy", cama_nextxy_1),
        #     input_field_1f,
        #     upstream_metric_min_1f,
        #     mv_1f,
        # ),
        # (
        #     ("cama_nextxy", cama_nextxy_1),
        #     input_field_1g,
        #     upstream_metric_min_1g,
        #     mv_1g,
        # ),
        # (
        #     ("cama_nextxy", cama_nextxy_2),
        #     input_field_2a,
        #     upstream_metric_min_2a,
        #     mv_2a,
        # ),
        # (
        #     ("cama_nextxy", cama_nextxy_2),
        #     input_field_2b,
        #     upstream_metric_min_2b,
        #     mv_2b,
        # ),
    ],
    indirect=["river_network"],
)
def test_calculate_upstream_metric_min(river_network, input_field, flow_downstream, mv):
    output_field = ekh.upstream.array.min(
        river_network, input_field, node_weights=None, return_type="masked"
    )
    print(output_field)
    print(flow_downstream)
    assert output_field.dtype == flow_downstream.dtype
    np.testing.assert_allclose(output_field, flow_downstream)


# # @pytest.mark.parametrize(
# #     "river_network, input_field, flow_downstream, mv",
# #     [
# #         (
# #             ("cama_nextxy", cama_nextxy_1),
# #             input_field_1a,
# #             upstream_metric_product_1a,
# #             mv_1a,
# #         ),
# #         (
# #             ("cama_nextxy", cama_nextxy_1),
# #             input_field_1b,
# #             upstream_metric_product_1b,
# #             mv_1b,
# #         ),
# #         (
# #             ("cama_nextxy", cama_nextxy_1),
# #             input_field_1c,
# #             upstream_metric_product_1c,
# #             mv_1c,
# #         ),
# #         (
# #             ("cama_nextxy", cama_nextxy_1),
# #             input_field_1d,
# #             upstream_metric_product_1d,
# #             mv_1d,
# #         ),
# #         (
# #             ("cama_nextxy", cama_nextxy_1),
# #             input_field_1e,
# #             upstream_metric_product_1e,
# #             mv_1e,
# #         ),
# #         (
# #             ("cama_nextxy", cama_nextxy_1),
# #             input_field_1f,
# #             upstream_metric_product_1f,
# #             mv_1f,
# #         ),
# #         (
# #             ("cama_nextxy", cama_nextxy_1),
# #             input_field_1g,
# #             upstream_metric_product_1g,
# #             mv_1g,
# #         ),
# #         (
# #             ("cama_nextxy", cama_nextxy_2),
# #             input_field_2a,
# #             upstream_metric_product_2a,
# #             mv_2a,
# #         ),
# #         (
# #             ("cama_nextxy", cama_nextxy_2),
# #             input_field_2b,
# #             upstream_metric_product_2b,
# #             mv_2b,
# #         ),
# #     ],
# #     indirect=["river_network"],
# # )
# # def test_calculate_upstream_metric_prod(
# #     river_network, input_field, flow_downstream, mv
# # ):
# #     output_field = ekh.upstream.array.prod(
# #         river_network,
# #         input_field,
# #         node_weights=None,
# #         mv=mv,
# #         accept_missing=True,
# #     )
# #     print(output_field)
# #     print(flow_downstream)
# #     assert output_field.dtype == flow_downstream.dtype
# #     np.testing.assert_allclose(output_field, flow_downstream)


@pytest.mark.parametrize(
    "river_network, input_field, flow_downstream, mv",
    [
        # (
        #     ("cama_nextxy", cama_nextxy_1),
        #     input_field_1a,
        #     upstream_metric_mean_1a,
        #     mv_1a,
        # ),
        # (
        #     ("cama_nextxy", cama_nextxy_1),
        #     input_field_1b,
        #     upstream_metric_mean_1b,
        #     mv_1b,
        # ),
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1c,
            upstream_metric_mean_1c,
            mv_1c,
        ),
        # (
        #     ("cama_nextxy", cama_nextxy_1),
        #     input_field_1d,
        #     upstream_metric_mean_1d,
        #     mv_1d,
        # ),
        (
            ("cama_nextxy", cama_nextxy_1),
            input_field_1e,
            upstream_metric_mean_1e,
            mv_1e,
        ),
        # (
        #     ("cama_nextxy", cama_nextxy_1),
        #     input_field_1f,
        #     upstream_metric_mean_1f,
        #     mv_1f,
        # ),
        # (
        #     ("cama_nextxy", cama_nextxy_1),
        #     input_field_1g,
        #     upstream_metric_mean_1g,
        #     mv_1g,
        # ),
        # (
        #     ("cama_nextxy", cama_nextxy_2),
        #     input_field_2a,
        #     upstream_metric_mean_2a,
        #     mv_2a,
        # ),
        # (
        #     ("cama_nextxy", cama_nextxy_2),
        #     input_field_2b,
        #     upstream_metric_mean_2b,
        #     mv_2b,
        # ),
    ],
    indirect=["river_network"],
)
def test_calculate_upstream_metric_mean(
    river_network, input_field, flow_downstream, mv
):
    output_field = ekh.upstream.array.mean(
        river_network, input_field, node_weights=None, return_type="masked"
    )
    assert output_field.dtype == flow_downstream.dtype
    np.testing.assert_allclose(output_field, flow_downstream)

    input_field = convert_to_2d(river_network, input_field, 0)
    flow_downstream = convert_to_2d(river_network, flow_downstream, 0)
    output_field = ekh.upstream.array.mean(
        river_network,
        input_field,
        node_weights=None,
    ).flatten()
    print(output_field)
    print(flow_downstream)
    assert output_field.dtype == flow_downstream.dtype
    np.testing.assert_allclose(output_field, flow_downstream)


# @pytest.mark.parametrize(
#     "river_network, input_field, accum_field",
#     [
#         (("d8_ldd", d8_ldd_1), input_field_1b, upstream_metric_sum_1g),
#         (("cama_downxy", cama_downxy_1), input_field_1b, upstream_metric_sum_1g),
#         (("cama_nextxy", cama_nextxy_1), input_field_1b, upstream_metric_sum_1g),
#         (("d8_ldd", d8_ldd_2), input_field_2b, upstream_metric_sum_2g),
#         (("cama_downxy", cama_downxy_2), input_field_2b, upstream_metric_sum_2g),
#         (("cama_nextxy", cama_nextxy_2), input_field_2b, upstream_metric_sum_2g),
#     ],
#     indirect=["river_network"],
# )
# def test_accumulate_downstream_missing(river_network, input_field, accum_field):
#     accum = ekh.upstream.array.sum(river_network, input_field, mv=-1, accept_missing=True)
#     print(accum)
#     print(accum_field)
#     np.testing.assert_array_equal(accum, accum_field)


# @pytest.mark.parametrize(
#     "river_network",
#     [
#         ("d8_ldd", d8_ldd_1),
#         ("cama_downxy", cama_downxy_1),
#         ("cama_nextxy", cama_nextxy_1),
#         ("d8_ldd", d8_ldd_2),
#         ("cama_downxy", cama_downxy_2),
#         ("cama_nextxy", cama_nextxy_2),
#     ],
#     indirect=True,
# )
# @pytest.mark.parametrize("N", range(4))
# def test_accumulate_downstream_2d(river_network, N):
#     field = np.random.rand(*([np.random.randint(10)] * N), *river_network.mask.shape)
#     field_1d = field[..., river_network.mask]
#     accum = ekh.upstream.array.sum(river_network, field_1d)
#     np.testing.assert_array_equal(
#         accum, ekh.upstream.array.sum(river_network, field)[..., river_network.mask]
#     )
#     np.testing.assert_array_equal(
#         ekh.upstream.array.sum(river_network, field)[..., ~river_network.mask],
#         field[..., ~river_network.mask],
#     )


# # @pytest.mark.parametrize(
# #     "river_network",
# #     [
# #         ("d8_ldd", d8_ldd_1),
# #         ("cama_downxy", cama_downxy_1),
# #         ("cama_nextxy", cama_nextxy_1),
# #         ("d8_ldd", d8_ldd_2),
# #         ("cama_downxy", cama_downxy_2),
# #         ("cama_nextxy", cama_nextxy_2),
# #     ],
# #     indirect=True,
# # )
# # @pytest.mark.parametrize("metric", ["sum", "max", "min", "mean", "std", "var"])
# # def test_calculate_catchment(river_network, metric):
# #     from earthkit.hydro.catchments.metrics import calculate_catchment_metric
# #     from earthkit.hydro.upstream.operations import calculate_upstream_metric

# #     field = np.random.rand(3, 4, river_network.n_nodes)
# #     node_weights = np.random.rand(3, 4, river_network.n_nodes)
# #     catchment_metric = calculate_catchment_metric(
# #         river_network, field, river_network.nodes, metric, node_weights
# #     )
# #     upstream_field = calculate_upstream_metric(
# #         river_network, field, metric, node_weights
# #     )
# #     print(catchment_metric[0].shape, upstream_field.shape)
# #     for i in catchment_metric.keys():
# #         assert catchment_metric[i].dtype == upstream_field.dtype
# #         np.testing.assert_allclose(catchment_metric[i], upstream_field[..., i])


# # @pytest.mark.parametrize(
# #     "river_network",
# #     [
# #         ("d8_ldd", d8_ldd_1),
# #         ("cama_downxy", cama_downxy_1),
# #         ("cama_nextxy", cama_nextxy_1),
# #         ("d8_ldd", d8_ldd_2),
# #         ("cama_downxy", cama_downxy_2),
# #         ("cama_nextxy", cama_nextxy_2),
# #     ],
# #     indirect=True,
# # )
# # @pytest.mark.parametrize("metric", ["sum", "max", "min", "mean", "prod", "std", "var"])
# # def test_calculate_subcatchment(river_network, metric):
# #     from earthkit.hydro.catchments.metrics import calculate_catchment_metric
# #     from earthkit.hydro.subcatchments.metrics import calculate_subcatchment_metric

# #     shape = (3, 4)
# #     field = np.random.rand(*shape, river_network.n_nodes) + 1
# #     node_weights = np.random.rand(*shape, river_network.n_nodes) + 1
# #     subcatchment_metric = calculate_subcatchment_metric(
# #         river_network, field, river_network.sinks, metric, node_weights
# #     )
# #     catchment_metric = calculate_catchment_metric(
# #         river_network, field, river_network.sinks, metric, node_weights
# #     )
# #     for i in catchment_metric.keys():
# #         assert catchment_metric[i].dtype == subcatchment_metric[i].dtype
# #         # the two methods don't match always because of numerical instabilities in the
# #         # variance computation arising from the different scales
# #         np.testing.assert_allclose(
# #             subcatchment_metric[i], catchment_metric[i], atol=5e-8
# #         )

# #     subcatchment_metric = calculate_subcatchment_metric(
# #         river_network, field, river_network.nodes, metric, node_weights
# #     )
# #     catchment_metric = calculate_catchment_metric(
# #         river_network,
# #         field,
# #         np.array(
# #             [
# #                 i
# #                 for i in river_network.nodes
# #                 if i not in river_network.sinks and i not in river_network.sources
# #             ]
# #         ),
# #         metric,
# #         node_weights,
# #     )
# #     for i in catchment_metric.keys():
# #         assert catchment_metric[i].dtype == subcatchment_metric[i].dtype
# #         np.testing.assert_raises(
# #             AssertionError,
# #             np.testing.assert_allclose,
# #             subcatchment_metric[i],
# #             catchment_metric[i],
# #         )


# # @pytest.mark.parametrize(
# #     "river_network",
# #     [
# #         ("d8_ldd", d8_ldd_1),
# #         ("cama_downxy", cama_downxy_1),
# #         ("cama_nextxy", cama_nextxy_1),
# #         ("d8_ldd", d8_ldd_2),
# #         ("cama_downxy", cama_downxy_2),
# #         ("cama_nextxy", cama_nextxy_2),
# #     ],
# #     indirect=True,
# # )
# # @pytest.mark.parametrize("metric", ["sum", "max", "min", "mean", "prod", "std", "var"])
# # def test_calculate_catchment_numpy(river_network, metric):
# #     from earthkit.hydro.catchments.metrics import calculate_catchment_metric

# #     shape = (3, 4)
# #     field = np.random.rand(*shape, river_network.n_nodes)
# #     node_weights = None
# #     nodes = np.random.choice(river_network.n_nodes, 5)
# #     catchment_metric = calculate_catchment_metric(
# #         river_network, field, nodes, metric, node_weights
# #     )
# #     numpy_func = get_numpy_function(metric)
# #     for i in catchment_metric.keys():
# #         mask = np.zeros(river_network.n_nodes, dtype=bool)
# #         mask[i] = True
# #         mask = ekh.catchments.find(river_network, mask)
# #         numpy_answer = numpy_func(field[..., mask], axis=-1)

# #         np.testing.assert_allclose(catchment_metric[i], numpy_answer)


# # @pytest.mark.parametrize(
# #     "river_network",
# #     [
# #         ("d8_ldd", d8_ldd_1),
# #         ("cama_downxy", cama_downxy_1),
# #         ("cama_nextxy", cama_nextxy_1),
# #         ("d8_ldd", d8_ldd_2),
# #         ("cama_downxy", cama_downxy_2),
# #         ("cama_nextxy", cama_nextxy_2),
# #     ],
# #     indirect=True,
# # )
# # @pytest.mark.parametrize("metric", ["sum", "max", "min", "mean", "prod", "std", "var"])
# # def test_calculate_subcatchment_numpy(river_network, metric):
# #     from earthkit.hydro.subcatchments.metrics import calculate_subcatchment_metric

# #     shape = (3, 4)
# #     field = np.random.rand(*shape, river_network.n_nodes)
# #     node_weights = None
# #     nodes = np.random.choice(river_network.n_nodes, 5, replace=False)
# #     subcatchment_metric = calculate_subcatchment_metric(
# #         river_network, field, nodes, metric, node_weights
# #     )
# #     numpy_func = get_numpy_function(metric)
# #     labels = np.zeros(river_network.n_nodes, dtype=int)
# #     labels[nodes] = nodes + 1
# #     labels = ekh.subcatchments.find(river_network, labels)
# #     for i in subcatchment_metric.keys():
# #         mask = labels == (i + 1)
# #         numpy_answer = numpy_func(field[..., mask], axis=-1)

# #         np.testing.assert_allclose(subcatchment_metric[i], numpy_answer)
