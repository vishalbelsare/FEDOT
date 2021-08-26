import pytest

from fedot.core.dag.validation_rules import has_no_cycle, has_no_isolated_components, has_no_isolated_nodes, \
    has_no_self_cycled_nodes
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.validation import (validate)
from fedot.core.pipelines.validation_rules import has_correct_operation_positions, has_final_operation_as_model, \
    has_no_conflicts_in_decompose, has_no_conflicts_with_data_flow, has_no_data_flow_conflicts_in_ts_pipeline, \
    has_primary_nodes, is_pipeline_contains_ts_operations, only_ts_specific_operations_are_primary, \
    has_correct_data_sources
from data.pipeline_manager import pipeline_with_cycle, valid_pipeline, pipeline_with_isolated_nodes,\
    pipeline_with_multiple_roots, pipeline_with_secondary_nodes_only, pipeline_with_self_cycle,\
    pipeline_with_isolated_components, pipeline_with_incorrect_parents_position_for_decompose,\
    pipeline_with_incorrect_task_type, pipeline_with_only_data_operations, pipeline_with_incorrect_data_flow,\
    ts_pipeline_with_incorrect_data_flow, pipeline_with_incorrect_parent_amount_for_decompose,\
    pipeline_with_incorrect_data_sources, pipeline_with_correct_data_sources

PIPELINE_ERROR_PREFIX = 'Invalid pipeline configuration:'
GRAPH_ERROR_PREFIX = 'Invalid graph configuration:'


def test_pipeline_with_cycle_raise_exception():
    pipeline = pipeline_with_cycle()
    with pytest.raises(Exception) as exc:
        assert has_no_cycle(pipeline)
    assert str(exc.value) == f'{GRAPH_ERROR_PREFIX} Graph has cycles'


def test_pipeline_without_cycles_correct():
    pipeline = valid_pipeline()

    assert has_no_cycle(pipeline)


def test_pipeline_with_isolated_nodes_raise_exception():
    pipeline = pipeline_with_isolated_nodes()
    with pytest.raises(ValueError) as exc:
        assert has_no_isolated_nodes(pipeline)
    assert str(exc.value) == f'{GRAPH_ERROR_PREFIX} Graph has isolated nodes'


def test_multi_root_pipeline_raise_exception():
    pipeline = pipeline_with_multiple_roots()

    with pytest.raises(Exception) as exc:
        assert pipeline.root_node
    assert str(exc.value) == f'{PIPELINE_ERROR_PREFIX} More than 1 root_nodes in pipeline'


def test_pipeline_with_primary_nodes_correct():
    pipeline = valid_pipeline()
    assert has_primary_nodes(pipeline)


def test_pipeline_without_primary_nodes_raise_exception():
    pipeline = pipeline_with_secondary_nodes_only()
    with pytest.raises(Exception) as exc:
        assert has_primary_nodes(pipeline)
    assert str(exc.value) == f'{PIPELINE_ERROR_PREFIX} Pipeline does not have primary nodes'


def test_pipeline_with_self_cycled_nodes_raise_exception():
    pipeline = pipeline_with_self_cycle()
    with pytest.raises(Exception) as exc:
        assert has_no_self_cycled_nodes(pipeline)
    assert str(exc.value) == f'{GRAPH_ERROR_PREFIX} Graph has self-cycled nodes'


def test_pipeline_validate_correct():
    pipeline = valid_pipeline()
    validate(pipeline)


def test_pipeline_with_isolated_components_raise_exception():
    pipeline = pipeline_with_isolated_components()
    with pytest.raises(Exception) as exc:
        assert has_no_isolated_components(pipeline)
    assert str(exc.value) == f'{GRAPH_ERROR_PREFIX} Graph has isolated components'


def test_pipeline_with_incorrect_task_type_raise_exception():
    pipeline, task = pipeline_with_incorrect_task_type()
    with pytest.raises(Exception) as exc:
        assert has_correct_operation_positions(pipeline, task)
    assert str(exc.value) == f'{PIPELINE_ERROR_PREFIX} Pipeline has incorrect operations positions'


def test_pipeline_without_model_in_root_node():
    incorrect_pipeline = pipeline_with_only_data_operations()

    with pytest.raises(Exception) as exc:
        assert has_final_operation_as_model(incorrect_pipeline)

    assert str(exc.value) == f'{PIPELINE_ERROR_PREFIX} Root operation is not a model'


def test_pipeline_with_incorrect_data_flow():
    incorrect_pipeline = pipeline_with_incorrect_data_flow()

    with pytest.raises(Exception) as exc:
        assert has_no_conflicts_with_data_flow(incorrect_pipeline)

    assert str(exc.value) == f'{PIPELINE_ERROR_PREFIX} Pipeline has incorrect subgraph with identical data operations'


def test_ts_pipeline_with_incorrect_data_flow():
    incorrect_pipeline = ts_pipeline_with_incorrect_data_flow()

    if is_pipeline_contains_ts_operations(incorrect_pipeline):
        with pytest.raises(Exception) as exc:
            assert has_no_data_flow_conflicts_in_ts_pipeline(incorrect_pipeline)

        assert str(exc.value) == \
               f'{PIPELINE_ERROR_PREFIX} Pipeline has incorrect subgraph with wrong parent nodes combination'
    else:
        assert False


def test_only_ts_specific_operations_are_primary():
    """ Incorrect pipeline
    lagged \
             linear -> final forecast
     ridge /
    """
    node_lagged = PrimaryNode('lagged')
    node_ridge = PrimaryNode('ridge')
    node_final = SecondaryNode('linear', nodes_from=[node_lagged, node_ridge])
    incorrect_pipeline = Pipeline(node_final)

    with pytest.raises(Exception) as exc:
        assert only_ts_specific_operations_are_primary(incorrect_pipeline)

    assert str(exc.value) == \
           f'{PIPELINE_ERROR_PREFIX} Pipeline for forecasting has not ts_specific preprocessing in primary nodes'


def test_has_two_parents_for_decompose_operations():
    incorrect_pipeline = pipeline_with_incorrect_parent_amount_for_decompose()

    with pytest.raises(Exception) as exc:
        assert has_no_conflicts_in_decompose(incorrect_pipeline)

    assert str(exc.value) == f'{PIPELINE_ERROR_PREFIX} Two parents for decompose node were expected, but 1 were given'


def test_decompose_parents_has_wright_positions():
    incorrect_pipeline = pipeline_with_incorrect_parents_position_for_decompose()

    with pytest.raises(Exception) as exc:
        assert has_no_conflicts_in_decompose(incorrect_pipeline)

    assert str(exc.value) == f'{PIPELINE_ERROR_PREFIX} For decompose operation Model as first parent is required'


def test_data_sources_validation():
    incorrect_pipeline = pipeline_with_incorrect_data_sources()

    with pytest.raises(ValueError) as exc:
        has_correct_data_sources(incorrect_pipeline)

    assert str(exc.value) == f'{PIPELINE_ERROR_PREFIX} Data sources are mixed with other primary nodes'

    correct_pipeline = pipeline_with_correct_data_sources()
    assert has_correct_data_sources(correct_pipeline)


def custom_validation_test():
    incorrect_pipeline = pipeline_with_incorrect_parents_position_for_decompose()

    assert validate(incorrect_pipeline, rules=[has_no_cycle])
