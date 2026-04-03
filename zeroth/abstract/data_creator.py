from abc import ABC, abstractmethod

from ..data import Data


class DataCreator(ABC):
    def __repr__(self):
        return f"{self.__class__.__name__}()"

    @abstractmethod
    def __call__(self) -> Data:
        pass
