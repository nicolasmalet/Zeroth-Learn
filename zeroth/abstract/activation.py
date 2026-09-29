from abc import ABC, abstractmethod

from ..types import Array


class Activation(ABC):
    """Base class for activation functions."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    @abstractmethod
    def __call__(self, x: Array) -> Array:
        """Apply the activation function."""
        ...

    @abstractmethod
    def derivative(self, x: Array) -> Array:
        """Compute the activation derivative."""
        ...
