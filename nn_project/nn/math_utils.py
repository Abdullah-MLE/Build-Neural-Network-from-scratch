"""Small mathematical operations shared by neural network components."""


def dot_product(first_values, second_values):
    """Return the sum of pairwise products from two lists."""
    # Multiply each value by the value at the same position.
    multiplied_values = [
        first_value * second_value
        for first_value, second_value in zip(first_values, second_values)
    ]

    # Add the products to get one final number.
    product_total = sum(multiplied_values)
    return product_total
