from .layer import Layer
from ..abstract.neural_network import NeuralNetwork, NeuralNetworkConfig

from ..types import Array


class FirstOrderNeuralNetwork(NeuralNetwork):
    """Standard Feed-Forward Neural Network consisting of a sequence of Layers.

    Attributes:
        layers (list[Layer]): Ordered list of layer objects.
        nb_layers (int): Number of layers.
    """

    def __init__(self, config: NeuralNetworkConfig) -> None:
        super().__init__(config)
        self.layers: list[Layer] = []
        for layer_config in config.layers_config:
            self.layers.append(Layer(layer_config.output_dim,
                                     layer_config.input_dim,
                                     layer_config.activation))

    def __call__(self, X: Array) -> Array:
        return self.forward(X)

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

    def forward(self, X: Array) -> Array:
        """Sequentially passes the input through all layers.

        Note:
            This method updates the internal state (self.X, self.Z, self.A) of each layer,
            which is required for the backward pass.
        """
        for layer in self.layers:
            X = layer(X)
        return X
