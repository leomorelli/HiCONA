"""Placeholder
Placeholder 
"""

from itertools import chain, combinations
from math import floor
from time import time

import numpy as np
from pandas import DataFrame
from scipy import integrate


# Store conversion dict since there is no automatic way to pass from ...
# ... pandas/numpy dtypes to graph-tool dtypes.
DTYPE_CONVERSION_DICT = {
    "int32": "int",
    "int64": "long",
    "float64": "long double",
    "object": "string",
}


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
        return int_cache[int_key], int_cache

    return integral_wrapper


@integration_cache
def compute_alpha_val(k: int, weight: float):
    """Compute alpha value according to Serrano et al. 2009."""
    int_func = lambda x, nn=k - 2: (1 - x) ** (nn)
    new_alpha = 1 - (k - 1) * integrate.quad(int_func, 0, weight)[0]

    return round_half_up(new_alpha, 4)


def from_df_to_sarrays(data: DataFrame):
    """Return each column of a dataframe as a numpy structured array.

    Transform the columns of a dataframe into numpy structured array and
    define the numpy datatype most appropriate for storage in HDF5 (especially
    minimum required string fixed length for categorical annotations).
    """
    dtypes = data.dtypes
    for col_name, dtype in zip(data, dtypes):
        if dtype == "object":
            str_len = int(data[col_name].str.len().max())
            str_len = str_len if str_len > 3 else 3
            dtype = np.dtype(f"S{str_len}")

        new_col = np.zeros(len(data), dtype)
        new_col[:] = data[col_name].values

        yield (col_name, new_col, dtype)


def pd_to_gt_dtype(pd_dtype: str):
    """Convert pandas-like datatypes to graph-tools datatypes"""
    return DTYPE_CONVERSION_DICT[pd_dtype]


def annotation_combinations(iterable, k_vals=(1, 2)):
    """Return iterable of all combinations for all k_vals"""

    comb = [combinations(iterable, k) for k in k_vals]
    return list(chain.from_iterable(comb))
