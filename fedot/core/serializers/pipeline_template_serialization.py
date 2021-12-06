from typing import Any, Dict

from fedot.core.pipelines.template import PipelineTemplate

from .any_serialization import any_to_json


def pipeline_template_to_json(obj: PipelineTemplate) -> Dict[str, Any]:
    return {
        k: v
        for k, v in any_to_json(obj).items()
        if k not in ['operation_templates', 'data_preprocessor']
    }
