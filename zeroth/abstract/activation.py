from abc import ABC, abstractmethod

from ..types import Array


class Activation(ABC):
    """Classe de base pour toutes les fonctions d'activation."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    @abstractmethod
    def __call__(self, x: float | Array) -> float | Array:
        """Applique la fonction d'activation (forward)."""
        pass

    @abstractmethod
    def derivative(self, x: float | Array) -> float | Array:
        """Calcule la dérivée de la fonction d'activation."""
        pass
