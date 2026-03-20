from typing import TypeAlias, Callable

import numpy as np

Array = np.typing.NDArray[np.float64]

ActivationFunction: TypeAlias = Callable[[float | Array], float | Array]
