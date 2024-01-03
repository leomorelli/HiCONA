"""Miscellaneous utility functions for the hicona package."""

from itertools import chain, combinations
import functools
from math import floor
import re
import time

from pybedtools import BedTool
from decorator import decorate
import h5py
import numpy as np
import pandas as pd
from scipy import integrate

from .settings import HICONA_SETTINGS


# Store conversion dict since there is no automatic way to pass from ...
# ... pandas/numpy dtypes to graph-tool dtypes.

# TODO: somehow add category


def console_log(func):
    """A simple decorator to log information to the console."""
    # TODO: Make decorator toggleable

    def console_log_wrapper(*args, **kwargs):
        print(f"Starting to run: {func}")
        start_time = time.time()
        fun_return = func(*args, **kwargs)
        end_time = time.time()
        print(f"Elapsed time: {end_time-start_time}s")
        print("-" * 78)
        return fun_return

    return console_log_wrapper


def wait_hdf5_lock(func):
    """Placeholder"""

    def _wait_hdf5_lock(func, *args, **kwargs):
        while True:
            try:
                fun_return = func(*args, **kwargs)
                break
            except BlockingIOError:
                time.sleep(HICONA_SETTINGS.parameters.lock_delay)

        return fun_return

    return decorate(func, _wait_hdf5_lock)


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

    @functools.wraps(func)
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


def from_df_to_sarrays(data: pd.DataFrame):
    """Return each column of a dataframe as a numpy structured array.

    Transform the columns of a dataframe into numpy structured array and
    define the numpy datatype most appropriate for storage in HDF5 (especially
    minimum required string fixed length for categorical annotations).
    """

    # TODO: Currently only discriminating string/non string, improve

    dtypes = data.dtypes
    for col_name, dtype in zip(data, dtypes):
        if dtype == "object":
            str_len = int(data[col_name].str.len().max())
            str_len = str_len if str_len > 3 else 3
            dtype = np.dtype(f"S{str_len}")

        new_col = np.zeros(len(data), dtype)
        new_col[:] = data[col_name].values

        yield (col_name, new_col, dtype)


def get_dataf_mapping(dataf: pd.DataFrame) -> dict:
    """Placeholder"""

    # Fetch dtype for each column name
    mapping = {k: v for k, v in zip(dataf.columns, dataf.dtypes)}

    # If dtype is object, convert to |SX where X is the max str length.
    # This is because hdf5 does not support object or generic string types.
    for k, v in mapping.items():
        if v == "object":
            conv_col = dataf[k].convert_dtypes()
            if conv_col.dtype == "string[python]":
                conv_col.fillna("nan", inplace=True)
                max_str_len = conv_col.map(len).max()
                mapping[k] = f"|S{max_str_len}"

            else:  # Cannot be converted to string
                raise TypeError("Object type annotations are not supported.")

    return mapping


def pd_to_h5_dtype(pd_dtype: str):
    """Stuff"""

    h5_dtype = pd_dtype
    if pd_dtype == "object":
        h5_dtype = np.dtype("S10")

    return h5_dtype


def pd_to_gt_dtype(pd_dtype: str):
    """Convert pandas-like datatypes to graph-tools datatypes"""
    return HICONA_SETTINGS.conventions.dtype_conversion[pd_dtype]


def annotation_combinations(iterable, k_vals=(1, 2)):
    """Return iterable of all combinations for all k_vals"""

    comb = [combinations(iterable, k) for k in k_vals]
    return list(chain.from_iterable(comb))


def parse_regions(regions, chroms):
    """Convert chromosome selection from regex/default str to iterable."""

    def fix_species_selection(selection):
        """Convert species selection string into list of chromosomes."""
        # "1-22,X,Y" -> ["chr1", "chr2", ..., "chr22", "chrX", "chrY"]

        out = []
        selection = selection.split(",")
        for s in selection:
            is_interval = len(s.split("-")) > 1
            if is_interval:
                lower, upper = s.split("-")
                for v in range(int(lower), int(upper) + 1):
                    out.append(f"chr{v}")
            else:
                out.append(f"chr{s}")

        return out

    intervals = []

    if isinstance(regions, str):
        regions = regions.strip()

        # "chrN:NNNN-NNNN" -> no formatting needed, return it
        if match := re.search(HICONA_SETTINGS.regexes.region, regions):
            intervals.append(regions)

        # "chrN" -> "chrN:NNNN-NNNN"
        elif match := re.search(HICONA_SETTINGS.regexes.chromosome, regions):
            c, s, e = chroms.query(f"chrom == '{match.group(0)}'").values[0].T
            intervals.append(f"{c}:{s}-{e}")

        # "organism" -> ["chrN:NNNN-NNNN", ...]
        elif match := HICONA_SETTINGS.conventions.chrom_lists.get(regions):
            for r in fix_species_selection(match):
                intervals.extend(parse_regions(r, chroms))

        # Incompatible string
        else:
            raise ValueError(f"{regions} is not a recognized genomic region.")

    # Assume it is an iterable of compatible objects
    else:
        for r in regions:
            intervals.extend(parse_regions(r, chroms))

    return intervals


def bedtool_to_dataframe(bed, colnames):
    """Placeholder"""

    if isinstance(bed, str):
        bed = BedTool(bed)

    num_columns = bed.field_count()
    if len(colnames) < num_columns:
        colnames += [None] * (num_columns - len(colnames))

    # Setting names after df creation to allow for duplicate colnames
    data = bed.to_dataframe(disable_auto_names=True, comment="#", header=None)
    data.columns = colnames

    # "." is the empty intersection for bedtools loj
    # Set after instead of using na_values="." due to type guessing issue
    data.replace(".", np.NaN, inplace=True)

    return data


def intersect_dataframes(df_a, df_b, **kwargs):
    """Return dataframe intersection using bedtools intersect.

    Placeholder
    """

    columns = list(df_a.columns) + list(df_b.columns)

    # TODO: Check handling of overlapping annotations in the same file
    # Suppress linting error due to pybedtools wrapper implementation
    # pylint: disable=unexpected-keyword-arg, too-many-function-args
    bed_a = BedTool.from_dataframe(df_a)
    bed_b = BedTool.from_dataframe(df_b)
    bed_a = bed_a.intersect(bed_b, **kwargs)
    # pylint: enable=unexpected-keyword-arg, too-many-function-args

    return bedtool_to_dataframe(bed_a, columns)
