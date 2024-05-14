"""Default functions for pixel table filtering."""

from typing import TYPE_CHECKING as _TYPE_CHECKING

import pandas as _pd

from hicona._dtypes import PdChunks as _PdChunks
from hicona._ops import chunked as _chunked

if _TYPE_CHECKING:
    from hicona._core import base_table


__all__ = [
    "filt_genomic_dist",
    "filt_column_quant",
    "filt_column_value",
    "filt_inter_chroms",
    "filt_self_looping",
]


def _inclusive_filter(
    table: _pd.DataFrame,
    column: str,
    lower: int | float | None = None,
    upper: int | float | None = None,
) -> _pd.DataFrame:
    """Remove pixels with column value outside of interval [lower, upper]."""

    table = table.loc[table[column] >= lower] if lower is not None else table
    table = table.loc[table[column] <= upper] if upper is not None else table

    return table


def _exclusive_filter(
    table: _pd.DataFrame,
    column: str,
    lower: int | float | None = None,
    upper: int | float | None = None,
) -> _pd.DataFrame:
    """Remove pixels with column value outside of interval (lower, upper)."""

    table = table.loc[table[column] > lower] if lower is not None else table
    table = table.loc[table[column] < upper] if upper is not None else table

    return table


def _format_out_cols(chunk: _pd.DataFrame) -> _pd.DataFrame:
    """Keep only bin1_id, bin2_id, count and norm (if present)."""

    base_cols = ["bin1_id", "bin2_id", "count", "norm"]
    keep_cols = [c for c in base_cols if c in chunk.columns]

    return chunk[keep_cols]


def filt_genomic_dist(
    table: "base_table.Table",
    min_dist: int | None = None,
    max_dist: int | None = None,
) -> _PdChunks:
    """Remove pixels whose genomic distance is outside of an interval.

    Genomic distance is calculated as the difference between bin1_id and
    bin2_id multiplied by the bin size (in bp). The pixels are removed if the
    distance is outside of the interval (extrema are kept).
    Genomic distance is computed only within the same chromosome.

    Parameters
    ----------
    table : Table
        The table to be filtered.
    min_dist : int or None, optional
        Remove pixels whose genomic distance greater than this value.
    max_dist : int or None, optional
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
        chunk.loc[chunk.chrom1 != chunk.chrom2, "dist"] = min_dist
        chunk = _inclusive_filter(chunk, "dist", min_dist, max_dist)

        yield _format_out_cols(chunk)


def filt_column_quant(
    table: "base_table.Table",
    apply_col: str,
    lower_quant: float | None = None,
    upper_quant: float | None = None,
    chrom_wise: bool = True,
) -> _PdChunks:
    """Remove pixels with column value outside a certain quantile range.

    Filter out pixels where the value of the specified column is outside a
    certain quantile threshold range (extrema are excluded).
    For each threshold, the corresponding value is computed, then all pixels
    with that value are removed. This means that if the bottom 10% pixels
    share the same value, asking to remove the bottom 1% will remove more all
    10%. This is done to avoid arbitrary tie breaks.

    Parameters
    ----------
    table : Table
        The table to be filtered.
    apply_col : str
        The column to apply the quantile filter to.
    lower_quant : float or None, optional
        Remove pixels whose column value is not greater than this quantile.
    upper_quant : float or None, optional
        Remove pixels whose column value is not smaller than this quantile.
    chrom_wise : bool, optional
        If True, compute quantiles for each pair of chromosomes separately.
        Default is True.

    Returns
    -------
    A generator of filtered pixel chunks.
    """

    def quant_filt(dataf, col, values, quant, sign, split_cols):
        """Apply individual quantile filter to a DataFrame."""

        values = values[values["quant"] == quant]
        if split_cols:
            merge = dataf.merge(values, how="left", on=split_cols)
            index = merge.query(f"{col}_x{sign}{col}_y").index
        else:
            quant_val = values[0][col]
            index = dataf.query(f"{col}{sign}{quant_val}").index
        return dataf.iloc[index]

    split_cols = ["chrom1", "chrom2"] if chrom_wise else None
    all_quants = [q for q in (lower_quant, upper_quant) if q is not None]

    if not any(all_quants):
        raise ValueError("At least one quantile threshold must be provided.")

    # Compute quantiles for each chromosome
    vals = _chunked.chunked_quants(
        table.chunks(annotated=chrom_wise),
        column=apply_col,
        quants=all_quants,
        split_on=split_cols,
    )

    # Apply the quantile filers to each chunk and yield it
    for chunk in table.chunks(annotated=chrom_wise):

        if lower_quant is not None:
            chunk = quant_filt(chunk, apply_col, vals, lower_quant, ">", split_cols)
        if upper_quant is not None:
            chunk = quant_filt(chunk, apply_col, vals, upper_quant, "<", split_cols)

        yield _format_out_cols(chunk)


def filt_column_value(
    table: "base_table.Table",
    apply_col: str,
    lower_value: float | int | None = None,
    upper_value: float | int | None = None,
) -> _PdChunks:
    """Remove pixels with column value outside a certain interval.

    Filter out pixels where the value of the specified column is outside a
    certain threshold (extrema are kept). Column must be numeric.

    Parameters
    ----------
    table : Table
        The table to be filtered.
    apply_col : str
        The column to apply the value filter to.
    lower_value : float or int or None, optional
        Remove pixels whose column value is not greater than this value.
    upper_value : float or int or None, optional
        Remove pixels whose column value is not smaller than this value.

    Returns
    -------
    A generator of filtered pixel chunks.
    """

    if not any([lower_value, upper_value]):
        raise ValueError("At least one value threshold must be provided.")

    for chunk in table.chunks():
        yield _inclusive_filter(chunk, apply_col, lower_value, upper_value)


def filt_inter_chroms(table: "base_table.Table") -> _PdChunks:
    """Remove inter-chromosomal pixels from the table.

    Remove pixels whose bin1_id and bin2_id are on different chromosomes.

    Parameters
    ----------
    table : Table
        The table to be filtered.

    Returns
    -------
    A generator of filtered pixel chunks.
    """

    for chunk in table.chunks(annotated=True):
        chunk = chunk.loc[chunk.chrom1 == chunk.chrom2]
        yield _format_out_cols(chunk)


def filt_self_looping(table: "base_table.Table") -> _PdChunks:
    """Remove self-looping pixels from the table.

    Remove pixels whose bin1_id and bin2_id are the same.

    Parameters
    ----------
    table : Table
        The table to be filtered.

    Returns
    -------
    A generator of filtered pixel chunks.
    """

    for chunk in table.chunks():
        chunk = chunk.loc[chunk.bin1_id != chunk.bin2_id]
        yield _format_out_cols(chunk)
