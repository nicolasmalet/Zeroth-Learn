from dataclasses import is_dataclass, fields, asdict, replace
from typing import Any


def get_name(obj: Any) -> Any:
    if hasattr(obj, "name"):
        return obj.name
    return obj


def generate_param_map(config_instance: Any, prefix: str = "", param_map: dict = None) -> dict:
    if param_map is None:
        param_map = {}

    if not is_dataclass(config_instance):
        return param_map

    for field in fields(config_instance):
        name = field.name

        path = f"{prefix}.{name}" if prefix else name

        if name not in param_map:
            param_map[name] = path
        elif name != "name":
            print(f"WARNING: attribute name {name} is used more than once in dataclasses")
        child = getattr(config_instance, name)

        if child is not None and is_dataclass(child):
            generate_param_map(child, path, param_map)

    return param_map


def get_catalog_values(catalog_instance: Any) -> list[Any]:
    return [getattr(catalog_instance, f.name) for f in fields(catalog_instance)]


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
