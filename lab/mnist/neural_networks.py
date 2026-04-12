import zeroth.utils.activation_functions as af
from zeroth.abstract import NeuralNetworkConfig

linear: NeuralNetworkConfig = NeuralNetworkConfig(
    name="linear",
    hidden_dims=[],
    activations=[af.Softmax()]
)

xs: NeuralNetworkConfig = NeuralNetworkConfig(
    name="xs",
    hidden_dims=[128],
    activations=[af.ReLU(),
                 af.Softmax()]

)

s: NeuralNetworkConfig = NeuralNetworkConfig(
    name="s",
    hidden_dims=[128, 64],
    activations=[af.ReLU(),
                 af.ReLU(),
                 af.Softmax()]
)

m: NeuralNetworkConfig = NeuralNetworkConfig(
    name="m",
    hidden_dims=[256, 128, 64],
    activations=[af.ReLU(),
                 af.ReLU(),
                 af.ReLU(),
                 af.Softmax()]
)

l: NeuralNetworkConfig = NeuralNetworkConfig(
    name="l",
    hidden_dims=[256, 256, 128, 64],
    activations=[af.ReLU(),
                 af.ReLU(),
                 af.ReLU(),
                 af.ReLU(),
                 af.Softmax()]
)
xl: NeuralNetworkConfig = NeuralNetworkConfig(
    name="xl",
    hidden_dims=[512, 512, 256, 128],
    activations=[af.ReLU(),
                 af.ReLU(),
                 af.ReLU(),
                 af.ReLU(),
                 af.ReLU(),
                 af.Softmax()]

)
