from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any


class Summary:
    def summary(self, file=None) -> None:
        print(self._summary(self, indent=0), file=file)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            print(self._summary(self, indent=0), file=f)

    @classmethod
    def load(cls, path: str, context: dict = None) -> Summary:
        with open(path, "r") as f:
            code = "".join(l for l in f)
        return eval(code.strip(), {"__builtins__": __builtins__}, context)

    def _summary(self, obj: Any, indent: int) -> str:
        shift = "    " * indent
        next_shift = "    " * (indent + 1)

        if dataclasses.is_dataclass(obj):
            cls_name = obj.__class__.__name__
            items = []
            for f in dataclasses.fields(obj):
                val = getattr(obj, f.name)
                formatted_val = self._summary(val, indent + 1)
                items.append(f"{next_shift}{f.name}={formatted_val.lstrip()}")

            content = ",\n".join(items)
            return f"{cls_name}(\n{content}\n{shift})"

        elif isinstance(obj, list):
            if not obj: return "[]"
            items = [self._summary(item, indent + 1) for item in obj]
            content = ",\n".join([f"{next_shift}{item.lstrip()}" for item in items])
            return f"[\n{content}\n{shift}]"

        elif isinstance(obj, dict):
            if not obj: return "{}"
            items = [f"{next_shift}{repr(k)}: {self._summary(v, indent + 1).lstrip()}" for k, v in obj.items()]
            content = ",\n".join(items)
            return f"{{\n{content}\n{shift}}}"

        else:
            return repr(obj)
