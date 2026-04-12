from dataclasses import replace
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


from dataclasses import is_dataclass, fields


def config_serializer(obj: Any):
    if is_dataclass(obj):
        data = {f.name: getattr(obj, f.name) for f in fields(obj)}
        return {obj.__class__.__name__: data}

    if hasattr(obj, "__class__") and obj.__class__.__module__ != "builtins":
        return repr(obj)

    if callable(obj):
        return getattr(obj, "__name__", str(obj))

    return str(obj)
