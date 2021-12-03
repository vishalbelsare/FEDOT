from typing import Any, Dict

from fedot.core.dag.graph_node import GraphNode

from . import any_serialization


def graph_node_to_json(obj: GraphNode) -> Dict[str, Any]:
    """
    Uses regular serialization but excludes "_operator" field to rid of circular references
    """
    return {
        k: v
        for k, v in any_serialization.any_to_json(obj).items()
        if k != '_operator'  # to prevent circular references
    }
