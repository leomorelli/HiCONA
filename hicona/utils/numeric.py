"""Placeholder"""

from math import floor
from scipy import integrate

from .decorators import integration_cache


def round_half_up(number: float, decimals: int = 0):
    """Return half way up rounded decimal number.

    Auxiliary function to round numbers since python default is not what it is
    commonly expected rounding to be. Half way up rounding means "round to
    closest value, either up or down, and break ties returning upper value".
    """
    multiplier: int = 10**decimals
    return floor(number * multiplier + 0.5) / multiplier


@integration_cache
def compute_alpha_val(k: int, weight: float):
    """Compute alpha value according to Serrano et al. 2009."""

    int_func = lambda x, nn=k - 2: (1 - x) ** (nn)
    new_alpha = 1 - (k - 1) * integrate.quad(int_func, 0, weight)[0]

    return round_half_up(new_alpha, 4)
