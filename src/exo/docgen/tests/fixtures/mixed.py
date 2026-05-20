def documented_function(x: int, y: int) -> int:
    """Add two numbers together.

    Args:
        x: First number.
        y: Second number.

    Returns:
        Sum of x and y.
    """
    return x + y


def undocumented_function(a, b):
    return a - b


class DocumentedClass:
    """A class with a documented docstring."""

    def __init__(self, value: int = 0):
        """Initialize the class.

        Args:
            value: Initial value.
        """
        self.value = value

    def undocumented_method(self, factor: int) -> int:
        return self.value * factor
