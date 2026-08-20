"""Neural network layers."""

import numpy


class Neuron:
    """A single neuron with a configurable activation function."""

    def __init__(self, weights, bias, activation_function):
        # Store the weights as a NumPy array so lists work with dot products.
        self.weights = numpy.array(weights, dtype=float)

        # Store the bias separately because it is added after the dot product.
        self.bias = float(bias)

        # Store the activation to call it after we calculate the value of z "pre activation".
        self.activation_function = activation_function

    def forward(self, inputs):
        # Convert the input list to the same numeric format as the weights.
        input_values = numpy.array(inputs, dtype=float)

        # Calculate the weighted sum of the input values.
        weighted_sum = numpy.dot(self.weights, input_values)

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
