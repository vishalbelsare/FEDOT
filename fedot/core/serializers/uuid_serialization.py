from typing import Any, Dict, Type
from uuid import UUID

from . import json_helpers


def uuid_to_json(obj: UUID) -> Dict[str, Any]:
    return {
        'hex': obj.hex,
        **json_helpers.dump_path_to_obj(obj)
    }


def uuid_from_json(cls: Type[UUID], json_obj: Dict[str, Any]) -> UUID:
    return cls(json_obj['hex'])
