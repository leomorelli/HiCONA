"""Default functions for pixel table normalization."""

import numpy as np

from ..hicona_table import RawTable
from ..utils.chunked_ops import groupwise_median
from ..utils.dtypes import PdChunks


__all__ = ["norm_genomic_dist", "norm_none", "norm_binwise"]


def norm_genomic_dist(table: RawTable, apply_col: str) -> PdChunks:
    """Apply default HiCONA normalization to a table (genomic distance).

    The normalized value is computed as the log2 of 1 plus the ratio of the
    value of the bin and some normalization factor. The normalization factor
    is computed as the median of the values of the bins at a given distance.

    Parameters
    ----------
    table: RawTable
        Pixel table to normalize.
    apply_col: str
        Column to apply the normalization to.

    Returns
    -------
    A generator of filtered pixel chunks.
    """

    def chunks_with_dist(table: RawTable):
        for chunk in table.chunks(annotated=True):
            chunk["diff"] = chunk.bin2_id - chunk.bin1_id
            yield chunk

    grp_cols = ["diff", "chrom1"]
    norm_curve = groupwise_median(chunks_with_dist(table), grp_cols, apply_col)
    norm_curve.rename("dist_norm", inplace=True)

    for chunk in chunks_with_dist(table):
        chunk = chunk.merge(norm_curve, how="left", on=grp_cols)
        chunk["norm"] = np.log2(chunk[apply_col] / chunk["dist_norm"] + 1)

        yield chunk[["bin1_id", "bin2_id", "count", "norm"]]


def norm_none(table: RawTable) -> PdChunks:
    """Apply no normalization to a table.

    Do not apply any normalization to the table, just return the original
    count values as the normalized values. This is used to copy the raw
    values to the normalized column.

    Parameters
    ----------
    table: RawTable
        Pixel table to normalize.

    Returns
    -------
    A generator of filtered pixel chunks.
    """

    for chunk in table.chunks():
        chunk["norm"] = chunk["count"]
        yield chunk[["bin1_id", "bin2_id", "count", "norm"]]


def norm_binwise(
    table: RawTable,
    apply_col: str,
    ann_name: str,
    divisive: bool = False,
    drop_nas: bool = False,
) -> PdChunks:
    """Apply a binwise normalization to a table.

    The normalization factors to use for normalization must be preemtively
    added to the cooler as a column in the bin table.

    Parameters
    ----------
    table: RawTable
        Pixel table to normalize.
    apply_col: str
        Column to apply the normalization to.
    ann_name: str
        Name of the column in the bin table to use for normalization.
    divisive: bool, optional
        If True, divide the values by the normalization factor, else multiply.
        Default is False.
    drop_nas: bool, optional
        If True, drop rows with NaN values after normalization.
        Default is False.

    Returns
    -------
    A generator of filtered pixel chunks.
    """

    # NOTE: This function mimicks matrix balancing normalization in cooler.

    col1, col2 = f"{ann_name}1", f"{ann_name}2"

    for chunk in table.chunks(annotated=True):
        if divisive:
            chunk[col1] = 1 / chunk[col1]
            chunk[col2] = 1 / chunk[col2]
        chunk["norm"] = chunk[apply_col] * chunk[col1] * chunk[col2]

        if drop_nas:
            chunk.dropna(inplace=True)

        yield chunk[["bin1_id", "bin2_id", "count", "norm"]]
