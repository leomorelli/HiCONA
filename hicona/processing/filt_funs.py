"""Pixel table filtering functions to be applied through Scheduler objects.

This module contains functions which are loaded by the `FiltScheduler` class
constructor and are thus able to be loaded in the pre-filtering or the
post-filtering schedulers during pixels table normalization. For these reason
the functions are not really mean for direct usage.

All functions share a common interface/architecture:
- a pixel table object must always be provided as first argument
- other arguments might be present (but not always)
- an iterator of processed pixel chunks is returned
- defult arguments should be avoided (they can make filtering opaque)

TODO: Only None as default
"""

from typing import Generator

import pandas as pd

from ..hicona_table import RawTable
from ..utils.chunked_ops import col_quants


__all__ = [
    "filter_genomic_dist",
    "filter_column_quant",
    "filter_column_value",
    "filter_inter_chroms",
]


PdChunks = Generator[pd.DataFrame, None, None]


def _inclusive_filter(
    table: pd.DataFrame,
    column: str,
    lower: int | float | None = None,
    upper: int | float | None = None,
) -> pd.DataFrame:
    """Remove pixels with column value outside of interval [lower, upper]."""

    table = table.loc[table[column] >= lower] if lower is not None else table
    table = table.loc[table[column] <= upper] if upper is not None else table

    return table


def _exclusive_filter(
    table: pd.DataFrame,
    column: str,
    lower: int | float | None = None,
    upper: int | float | None = None,
) -> pd.DataFrame:
    """Remove pixels with column value outside of interval (lower, upper)."""

    table = table.loc[table[column] > lower] if lower is not None else table
    table = table.loc[table[column] < upper] if upper is not None else table

    return table


def filter_genomic_dist(
    table: RawTable,
    min_dist: int | None = None,
    max_dist: int | None = None,
) -> PdChunks:
    """Remove pixels whose genomic distance is outside of an interval.

    Genomic distance is calculated as the difference between bin1_id and
    bin2_id multiplied by the bin size (in bp). The pixels are removed if the
    distance is outside of the interval (extrema are kept).
    Genomic distance is computed only within the same chromosome.

    Parameters
    -----------
    table: RawTable
        The table to be filtered.
    min_dist: int or None, optional
        Remove pixels whose genomic distance greater than this value.
    max_dist: int or None, optional
        Remove pixels whose genomic distance smaller than this value.

    Returns
    -------
    A generator of filtered pixel chunks.
    """

    if not any([min_dist, max_dist]):
        raise ValueError("At least one distance threshold must be provided.")

    # Workaround: set inter-chromosomal distances to set value in interval
    for chunk in table.chunks(annotated=True):
        chunk["dist"] = (chunk.bin2_id - chunk.bin1_id) * table.bin_size
        chunk.loc[chunk.chrom1 == chunk.chrom2, "dist"] = min_dist
        chunk = _inclusive_filter(chunk, "dist", min_dist, max_dist)
        yield chunk[["bin1_id", "bin2_id", "count"]]


def filter_column_quant(
    table: RawTable,
    apply_col: str,
    lower_quant: float | None = None,
    upper_quant: float | None = None,
) -> PdChunks:
    """Remove pixels with column value outside a certain quantile range.

    Filter out pixels where the value of the specified column is outside a
    certain quantile threshold range (extrema are excluded).
    For each threshold, the corresponding value is computed, then all pixels
    with that value are removed. This means that if the bottom 10% pixels
    share the same value, asking to remove the bottom 1% will remove more all
    10%. This is done to avoid arbitrary tie breaks.

    Parameters
    -----------
    table: RawTable
        The table to be filtered.
    apply_col: str
        The column to apply the quantile filter to.
    lower_quant: float or None, optional
        Remove pixels whose column value is not greater than this quantile.
    upper_quant: float or None, optional
        Remove pixels whose column value is not smaller than this quantile.

    Returns
    -------
    A generator of filtered pixel chunks.
    """

    if not any([lower_quant, upper_quant]):
        raise ValueError("At least one quantile threshold must be provided.")

    lo, hi = lower_quant, upper_quant
    lo = None if lo is None else col_quants(table.chunks(), apply_col, lo)[0]
    hi = None if hi is None else col_quants(table.chunks(), apply_col, hi)[0]

    for chunk in table.chunks():
        yield _exclusive_filter(chunk, apply_col, lo, hi)


def filter_column_value(
    table: RawTable,
    apply_col: str,
    lower_value: float | int | None = None,
    upper_value: float | int | None = None,
) -> PdChunks:
    """Remove pixels with column value outside a certain interval.

    Filter out pixels where the value of the specified column is outside a
    certain threshold (extrema are kept). Column must be numeric.

    Parameters
    -----------
    table: RawTable
        The table to be filtered.
    apply_col: str
        The column to apply the value filter to.
    lower_value: float or int or None, optional
        Remove pixels whose column value is not greater than this value.
    upper_value: float or int or None, optional
        Remove pixels whose column value is not smaller than this value.

    Returns
    -------
    A generator of filtered pixel chunks.
    """

    if not any([lower_value, upper_value]):
        raise ValueError("At least one value threshold must be provided.")

    for chunk in table.chunks():
        yield _inclusive_filter(chunk, apply_col, lower_value, upper_value)


def filter_inter_chroms(table: RawTable) -> PdChunks:
    """Remove inter-chromosomal pixels from the table.

    Remove pixels whose bin1_id and bin2_id are on different chromosomes.

    Parameters
    -----------
    table: RawTable
        The table to be filtered.

    Returns
    -------
    A generator of filtered pixel chunks.
    """

    for chunk in table.chunks(annotated=True):
        chunk = chunk.loc[chunk.chrom1 == chunk.chrom2]
        yield chunk[["bin1_id", "bin2_id", "count"]]
