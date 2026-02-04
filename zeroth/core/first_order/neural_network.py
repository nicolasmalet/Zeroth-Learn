import numpy as np

from zeroth.core.abstract.neural_network import NeuralNetwork, NeuralNetworkConfig
from zeroth.core.first_order.layer import Layer


class FirstOrderNeuralNetwork(NeuralNetwork):
    """Standard Feed-Forward Neural Network consisting of a sequence of Layers.

    Attributes:
        layers (list[Layer]): Ordered list of layer objects.
        nb_layers (int): Number of layers.
    """

    def __init__(self, config: NeuralNetworkConfig):
        self.name = config.name
        self.layers = []
        self.nb_layers = 0
        for layer_config in config.layers_config:
            self.layers.append(Layer(layer_config.output_dim,
                                     layer_config.input_dim,
                                     layer_config.f))
            self.nb_layers += 1

    def init_params(self, Ws, Bs):
        for layer, W, B in zip(self.layers, Ws, Bs):
            layer.W = W
            layer.B = B

    def get_params(self):
        Ws, Bs = [], []
        for layer in self.layers:
            Ws.append(layer.W)
            Bs.append(layer.B)
        return Ws, Bs

    def print_params(self):
        Ws, Bs = self.get_params()
        print(f"Ws : {Ws[:10]}, Bb : {Bs[:10]}")

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Sequentially passes the input through all layers.

        Note:
            This method updates the internal state (self.X, self.Z, self.A) of each layer,
            which is required for the backward pass.
        """
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def show_weights(self):
        Ws, Bs = self.get_params()
        return f"Ws, {Ws}, Bs {Bs}"
