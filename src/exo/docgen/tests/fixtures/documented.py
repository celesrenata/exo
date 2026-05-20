"""Module for a simple calculator.

This module provides basic arithmetic operations.
"""


def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First integer to add.
        b: Second integer to add.

    Returns:
        The sum of a and b.
    """
    return a + b


class Calculator:
    """A simple calculator class.

    Attributes:
        result: Current result of calculations.
    """

    def __init__(self, initial: int = 0) -> None:
        """Initialize the calculator with an optional initial value.

        Args:
            initial: Starting value for the calculator. Defaults to 0.
        """
        self.result = initial

    def multiply(self, value: int) -> int:
        """Multiply the current result by a given value.

        Args:
            value: The multiplier.

        Returns:
            The new result after multiplication.
        """
        self.result *= value
        return self.result
