"""Default functions for pixel table normalization."""

from typing import TYPE_CHECKING as _TYPE_CHECKING

import numpy as _np

from hicona._dtypes import PdChunks as _PdChunks
from hicona._ops import chunked as _chunked

if _TYPE_CHECKING:
    from hicona._core import base_table


__all__ = ["norm_genomic_dist", "norm_none", "norm_binwise"]


def norm_genomic_dist(table: "base_table.Table", apply_col: str) -> _PdChunks:
    """Apply default HiCONA normalization to a table (genomic distance).

    The normalized value is computed as the log2 of 1 plus the ratio of the
    value of the bin and some normalization factor. The normalization factor
    is computed as the median of the values of the bins at a given distance.
    Raises and error if inter-chromosomal pixels are found.

    Parameters
    ----------
    table : Table
        Pixel table to normalize.
    apply_col : str
        Column to apply the normalization to.

    Returns
    -------
    A generator of filtered pixel chunks.
    """

    def distance_iter(table_obj):
        """Iter chunks with genomic distance. Add inter-chromosomal check."""

        chunks = table_obj.chunks(annotated=True)
        bin_size = table_obj.bin_size

        for chunk in _chunked.add_gen_dist(chunks, bin_size):

            # Check that inter-chromosomal pixels where removed
            if any(chunk["chrom1"] != chunk["chrom2"]):
                raise ValueError("Inter-chromosomal pixels must be removed.")

            yield chunk

    norm_curve = _chunked.chunked_quants(
        distance_iter(table),
        column=apply_col,
        quants=0.5,
        split_on=["chrom1", "chrom2"],
        group_by="dist",
    )
    norm_curve.rename(columns={apply_col: "dist_norm"}, inplace=True)

    for chunk in distance_iter(table):
        chunk = chunk.merge(norm_curve, how="left", on=["chrom1", "dist"])
        chunk["norm"] = _np.log2(chunk[apply_col] / chunk["dist_norm"] + 1)

        yield chunk[["bin1_id", "bin2_id", "count", "norm"]]


def norm_none(table: "base_table.Table") -> _PdChunks:
    """Apply no normalization to a table.

    Do not apply any normalization to the table, just return the original
    count values as the normalized values. This is used to copy the raw
    values to the normalized column.

    Parameters
    ----------
    table : Table
        Pixel table to normalize.

    Returns
    -------
    A generator of filtered pixel chunks.
    """

    for chunk in table.chunks():
        chunk["norm"] = chunk["count"]
        yield chunk[["bin1_id", "bin2_id", "count", "norm"]]


def norm_binwise(
    table: "base_table.Table",
    apply_col: str,
    ann_name: str,
    divisive: bool = False,
    drop_nas: bool = False,
) -> _PdChunks:
    """Apply a binwise normalization to a table.

    The normalization factors to use for normalization must be preemtively
    added to the cooler as a column in the bin table.

    Parameters
    ----------
    table : Table
        Pixel table to normalize.
    apply_col : str
        Column to apply the normalization to.
    ann_name : str
        Name of the column in the bin table to use for normalization.
    divisive : bool, optional
        If True, divide the values by the normalization factor, else multiply.
        Default is False.
    drop_nas : bool, optional
        If True, drop rows with NaN values after normalization.
        Default is False.

    Returns
    -------
    A generator of filtered pixel chunks.
    """

    # NOTE: This function mimics matrix balancing normalization in cooler.

    col1, col2 = f"{ann_name}1", f"{ann_name}2"

    for chunk in table.chunks(annotated=True):
        if divisive:
            chunk[col1] = 1 / chunk[col1]
            chunk[col2] = 1 / chunk[col2]
        chunk["norm"] = chunk[apply_col] * chunk[col1] * chunk[col2]

        if drop_nas:
            chunk.dropna(inplace=True)

        yield chunk[["bin1_id", "bin2_id", "count", "norm"]]
