"""Neural network layers."""

from .math_utils import dot_product


class Neuron:
    """A single neuron with a configurable activation function."""

    def __init__(self, weights, bias, activation_function):
        # Store the weights as a list of numbers.
        self.weights = [float(weight) for weight in weights]

        # Store the bias separately because it is added after the dot product.
        self.bias = float(bias)

        # Store the activation to call it after we calculate the value of z "pre activation".
        self.activation_function = activation_function

    def forward(self, inputs):
        # Calculate the weighted sum using the dot product helper.
        weighted_sum = dot_product(self.weights, inputs)

        # Add the bias to complete the neuron's linear calculation.
        neuron_value = weighted_sum + self.bias

        # Apply the selected activation function to produce the final output.
        output_value = self.activation_function(neuron_value)
        return output_value


class Dense:
    pass


class Conv2D:
    pass


class MaxPool2D:
    pass


class Embedding:
    pass
