import json

from ..utils.dataclasses_utils import config_serializer


class Summary:
    def summary(self) -> None:
        summary = json.dumps(
            self,
            default=config_serializer,
            indent=4
        )
        print(summary)
