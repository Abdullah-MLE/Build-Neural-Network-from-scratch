"""Activation functions and their derivatives."""

import math


def sigmoid(input_value: float) -> float:
    # Sigmoid converts any number into a value between 0 and 1.
    # This is the sigmoid formula: 1 / (1 + e^(-input_value)).
    exponential_value = math.exp(-input_value)
    sigmoid_value = 1.0 / 1.0 + exponential_value
    return sigmoid_value


def sigmoid_backward(grad, x):
    pass


def relu(x):
    pass


def relu_backward(grad, x):
    pass


def softmax(x):
    pass


def softmax_backward(grad, x):
    pass
