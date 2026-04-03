from dataclasses import is_dataclass, asdict, replace
from typing import Any


def get_name(obj: Any) -> Any:
    if hasattr(obj, "name"):
        return obj.name
    return obj

def set_value_by_path(obj: Any, path: str, value: Any) -> Any:
    parts = path.split(".", 1)
    field = parts[0]

    is_seq = field.isdigit() and isinstance(obj, (list, tuple))

    if len(parts) > 1:
        child = obj[int(field)] if is_seq else getattr(obj, field)
        value = set_value_by_path(child, parts[1], value)

    if is_seq:
        new_seq = list(obj)
        new_seq[int(field)] = value
        return type(obj)(new_seq)

    return replace(obj, **{field: value})


def config_serializer(obj: Any):
    if is_dataclass(obj):
        return asdict(obj)
    if callable(obj):
        return getattr(obj, "__name__", str(obj))
    if hasattr(obj, "name"):
        return obj.name
    return str(obj)
