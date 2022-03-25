from itertools import product
from typing import Any, Dict, Type

from fedot.core.dag.graph import Graph

from . import any_to_json


def graph_to_json(obj: Graph) -> Dict[str, Any]:
    """
    Uses regular serialization but excludes "operator" field to rid of circular references
        also saves idx of each node from 'nodes' field to simplify deserialization
    """
    serialized_obj = {
        k: v
        for k, v in any_to_json(obj).items()
        if k != 'operator'  # to prevent circular reference
    }
    for idx, node in enumerate(serialized_obj['nodes']):
        node._serialization_id = idx
        node.num_of_children = node.calculate_num_of_children(obj)
        node.distance_to_root_level = node.calculate_distance_to_root_level(obj)
        child_nodes = [n for n in obj.nodes if n.nodes_from and node in n.nodes_from]
        # Num of brothers is length of parents nodes for all child nodes - length of child nodes
        node.num_of_brothers = sum([len(n.nodes_from) for n in child_nodes if n.nodes_from]) - len(child_nodes)
    return serialized_obj


def graph_from_json(cls: Type[Graph], json_obj: Dict[str, Any]) -> Graph:
    """
    Assigns each <inner_node> from "nodes_from" to equal <outer_node> from "nodes"
        (cause each node from "nodes_from" in fact should point to the same node from "nodes")
    """
    obj = cls()
    nodes = json_obj['nodes']
    for node in nodes:
        if node.nodes_from:
            for (idx, inner_node_idx), outer_node in product(enumerate(node.nodes_from), nodes):
                if inner_node_idx == outer_node._serialization_id:
                    node.nodes_from[idx] = outer_node
    obj.nodes = nodes
    vars(obj).update(**{k: v for k, v in json_obj.items() if k != 'nodes'})
    return obj
