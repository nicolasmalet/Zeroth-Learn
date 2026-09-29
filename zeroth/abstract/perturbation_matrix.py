from abc import ABC, abstractmethod

from ..types import Array


class PerturbationMatrix(ABC):
    """Base class for perturbation-matrix generators."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    @abstractmethod
    def __call__(self, nb_perturbation: int, nb_parameters: int) -> Array:
        """Generate perturbation directions."""
        ...
