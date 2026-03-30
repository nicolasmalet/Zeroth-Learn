from typing import TypeAlias, Callable

import numpy as np

Array = np.typing.NDArray

ActivationFunction: TypeAlias = Callable[[float | Array], float | Array]
