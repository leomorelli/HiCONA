"""Default classes for pixel table normalization."""

from typing import TYPE_CHECKING as _TYPE_CHECKING

import polars as pl

from hicona._ops import chunked
from hicona.preprocess._abcs import NormOperation
from hicona.preprocess._anno import add_annot, add_genomic_dist

if _TYPE_CHECKING:
    from hicona._core import PixelTable
    from hicona._dtypes import DfChunks


__all__ = ["NormBinwise", "NormGenomicDist"]


class NormBinwise(NormOperation):
    """Apply a binwise normalization to a table.

    The normalization factors to use for normalization must be preemtively
    added to the cooler file as a column in the bin table.

    Parameters
    ----------
    apply_col : str
        Table column to apply the normalization to.
    ann_name : str
        Name of the column in the bin table to use for normalization.
    divisive : bool, optional
        If True, divide the values by the normalization factor, else multiply.
        Default is False.
    drop_nas : bool, optional
        If True, drop rows with NaN values after normalization.
        Default is False.

    Notes
    -----
    This function mimics matrix balancing normalization found in :mod:`cooler`.
    """

    def __init__(
        self,
        *,
        apply_col: str,
        ann_name: str,
        divisive: bool = False,
        drop_nas: bool = False,
    ):
        self._apply_col = apply_col
        self._ann_name = ann_name
        self._divisive = divisive
        self._drop_nas = drop_nas

    def process(self, table: "PixelTable") -> "DfChunks":

        col1, col2 = f"{self._ann_name}1", f"{self._ann_name}2"

        for chunk in add_annot(table, self._ann_name):

            if self._divisive:
                chunk = chunk.with_columns(1 / pl.col(col1), 1 / pl.col(col2))

            exp = (pl.col(self._apply_col) * pl.col(col1) * pl.col(col2)).alias("norm")

            df = chunk.with_columns(exp).drop(col1, col2)
            df = df.drop_nulls() if self._drop_nas else df

            yield df


class NormGenomicDist(NormOperation):
    """Apply default HiCONA normalization to a table (genomic distance).

    The normalized value is computed as the log2 of 1 plus the ratio of the
    value of the bin and some normalization factor. The normalization factor
    is computed as the mean of the values of the bins at a given distance.
    Raises and error if inter-chromosomal pixels are found.
    """

    def __init__(self, *, apply_col: str):
        self._apply_col = apply_col

    def process(self, table: "PixelTable") -> "DfChunks":

        chunks = chunked.chunked_groupby(
            add_genomic_dist(table), split_on=["chrom1", "chrom2"]
        )

        for chunk in chunks:
            chunk = (
                chunk.with_columns(
                    pl.col(self._apply_col).mean().over("dist").alias("norm_factor")
                )
                .with_columns(
                    (pl.col(self._apply_col) / pl.col("norm_factor")).alias("norm")
                )
                .drop("dist", "chrom1", "chrom2", "norm_factor")
            )
            yield chunk
