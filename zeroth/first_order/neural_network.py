import numpy as np

from .layer import Layer
from ..abstract.blackbox import BlackBox, NeuralNetworkConfig


class FirstOrderNeuralNetwork(BlackBox):
    """Standard Feed-Forward Neural Network consisting of a sequence of Layers.

    Attributes:
        layers (list[Layer]): Ordered list of layer objects.
        nb_layers (int): Number of layers.
    """

    def __init__(self, config: NeuralNetworkConfig) -> None:
        self.name: str = config.name
        self.layers: list[Layer] = []
        self.nb_layers: int = 0
        self.input_dim: int = config.layers_config[0].input_dim
        self.output_dim: int = config.layers_config[-1].output_dim
        for layer_config in config.layers_config:
            self.layers.append(Layer(layer_config.output_dim,
                                     layer_config.input_dim,
                                     layer_config.f))
            self.nb_layers += 1

    def init_params(self, params: dict) -> None:
        Ws = params["Ws"]
        Bs = params["Bs"]
        for layer, W, B in zip(self.layers, Ws, Bs):
            layer.W = W
            layer.B = B

    def get_params(self) -> dict:
        Ws, Bs = [], []
        for layer in self.layers:
            Ws.append(layer.W)
            Bs.append(layer.B)
        return {"Ws": Ws, "Bs": Bs}

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Sequentially passes the input through all layers.

        Note:
            This method updates the internal state (self.X, self.Z, self.A) of each layer,
            which is required for the backward pass.
        """
        for layer in self.layers:
            X = layer.forward(X)
        return X
