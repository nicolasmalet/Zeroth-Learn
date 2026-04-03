import zeroth.utils.activation_functions as af
from zeroth.abstract import LayerConfig, NeuralNetworkConfig
from .task_constants import DATA_SIZE, NB_CLASS

linear: NeuralNetworkConfig = NeuralNetworkConfig(
    name="Linear",
    layers_config=[LayerConfig(input_dim=DATA_SIZE, output_dim=NB_CLASS, activation=af.Softmax())]
)

xs: NeuralNetworkConfig = NeuralNetworkConfig(
    name="XS",
    layers_config=[
        LayerConfig(input_dim=DATA_SIZE, output_dim=128, activation=af.ReLU()),
        LayerConfig(input_dim=128, output_dim=NB_CLASS, activation=af.Softmax())
    ]
)

s: NeuralNetworkConfig = NeuralNetworkConfig(
    name="S",
    layers_config=[
        LayerConfig(input_dim=DATA_SIZE, output_dim=128, activation=af.ReLU()),
        LayerConfig(input_dim=128, output_dim=64, activation=af.ReLU()),
        LayerConfig(input_dim=64, output_dim=NB_CLASS, activation=af.Softmax())
    ]
)

m: NeuralNetworkConfig = NeuralNetworkConfig(
    name="M",
    layers_config=[
        LayerConfig(input_dim=DATA_SIZE, output_dim=256, activation=af.ReLU()),
        LayerConfig(input_dim=256, output_dim=128, activation=af.ReLU()),
        LayerConfig(input_dim=128, output_dim=64, activation=af.ReLU()),
        LayerConfig(input_dim=64, output_dim=NB_CLASS, activation=af.Softmax())
    ]
)

l: NeuralNetworkConfig = NeuralNetworkConfig(
    name="L",
    layers_config=[
        LayerConfig(input_dim=DATA_SIZE, output_dim=256, activation=af.ReLU()),
        LayerConfig(input_dim=256, output_dim=256, activation=af.ReLU()),
        LayerConfig(input_dim=256, output_dim=128, activation=af.ReLU()),
        LayerConfig(input_dim=128, output_dim=NB_CLASS, activation=af.Softmax())
    ]
)
xl: NeuralNetworkConfig = NeuralNetworkConfig(
    name="XL",
    layers_config=[
        LayerConfig(input_dim=DATA_SIZE, output_dim=512, activation=af.ReLU()),
        LayerConfig(input_dim=512, output_dim=512, activation=af.ReLU()),
        LayerConfig(input_dim=512, output_dim=256, activation=af.ReLU()),
        LayerConfig(input_dim=256, output_dim=128, activation=af.ReLU()),
        LayerConfig(input_dim=128, output_dim=NB_CLASS, activation=af.Softmax())
    ]
)
