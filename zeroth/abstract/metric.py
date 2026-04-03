from abc import ABC, abstractmethod

from ..types import Array


class Metric(ABC):
    def __repr__(self):
        return f"{self.__class__.__name__}()"

    @abstractmethod
    def __call__(self, Y_pred: Array, Y_true: Array) -> float:
        """Applique la fonction d'activation (forward)."""
        pass
