"""Functions dealing with numeric operations."""

from math import floor


def round_half_up(number: float, decimals: int = 0):
    """Return half way up rounded decimal number.

    Auxiliary function to round numbers since python default is not what it is
    commonly expected rounding to be. Half way up rounding means "round to
    closest value, either up or down, and break ties returning upper value".
    """
    multiplier: int = 10**decimals
    return floor(number * multiplier + 0.5) / multiplier
