from typing import Any, Dict

from fedot.core.dag.graph_node import GraphNode

from . import any_to_json


def graph_node_to_json(obj: GraphNode) -> Dict[str, Any]:
    """
    Uses regular serialization but excludes "_operator" field to rid of circular references
    """
    encoded = {
        k: v
        for k, v in any_to_json(obj).items()
        if k not in ['_operator', '_fitted_operation', '_node_data']
    }
    encoded['content']['name'] = str(encoded['content']['name'])
    encoded['distance_to_primary_level'] = obj.distance_to_primary_level
    encoded['distance_to_root_level'] = obj.distance_to_root_level
    encoded['num_of_parents'] = obj.num_of_parents
    encoded['num_of_children'] = obj.num_of_children
    if encoded['nodes_from']:
        encoded['nodes_from'] = [
            node._serialization_id
            for node in encoded['nodes_from']
        ]
    return encoded
