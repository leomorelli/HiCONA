"""Placeholder
Placeholder 
"""

from math import floor
from time import time

from scipy import integrate


def console_log(func):
    """A simple decorator to log information to the console."""
    # TODO: Make decorator toggleable

    def console_log_wrapper(*args, **kwargs):
        print(f"Starting to run: {func}")
        start_time = time()
        fun_return = func(*args, **kwargs)
        end_time = time()
        print(f"Elapsed time: {end_time-start_time}s")
        print("-" * 79)
        return fun_return

    return console_log_wrapper


def round_half_up(number: float, decimals: int = 0):
    """Return half way up rounded decimal number.

    Auxiliary function to round numbers since python default is not what it is
    commonly expected rounding to be. Half way up rounding means "round to
    closest value, either up or down, and break ties returning upper value".
    """
    multiplier = 10**decimals
    return floor(number * multiplier + 0.5) / multiplier


def integration_cache(func):
    """Decorator to memoize alpha value integrals."""
    # TODO: make decorator toggleable
    int_cache = {}

    def integral_wrapper(*args, **kwargs):
        int_key = str(args) + str(kwargs)
        if int_key not in int_cache:
            int_cache[int_key] = func(*args, **kwargs)
        return int_cache[int_key]

    return integral_wrapper


@integration_cache
def compute_alpha_val(k: int, weight: float):
    """Compute alpha value according to Serrano et al. 2009."""
    int_func = lambda x, nn=k - 2: (1 - x) ** (nn)
    new_alpha = 1 - (k - 1) * integrate.quad(int_func, 0, weight)[0]

    return round_half_up(new_alpha, 4)
