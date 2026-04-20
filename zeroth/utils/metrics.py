from typing import Self

import numpy as np

from ..abstract.metric import Metric
from ..types import Array


class Accuracy(Metric):
    def __call__(self: Self, Y_pred: Array, Y_true: Array) -> float:
        return float(np.mean(np.argmax(Y_pred, axis=1) == Y_true[:, 0]))
