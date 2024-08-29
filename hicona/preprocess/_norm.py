"""Default classes for pixel table normalization."""

from typing import TYPE_CHECKING as _TYPE_CHECKING

import polars as pl

from hicona._ops import chunked
from hicona.preprocess._abcs import NormOperation

if _TYPE_CHECKING:
    from hicona._core import Table
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

    def process(self, table: "Table") -> "DfChunks":

        col1, col2 = f"{self._ann_name}1", f"{self._ann_name}2"

        for chunk in table.chunks(annotated=True):

            if self._divisive:
                chunk = chunk.with_columns(1 / pl.col(col1), 1 / pl.col(col2))

            exp = (pl.col(self._apply_col) * pl.col(col1) * pl.col(col2)).alias("norm")
            exp = exp.drop_nulls() if self._drop_nas else exp

            yield chunk.with_columns(exp)


# class NormGenomicDist(NormOperation):
#     """Apply default HiCONA normalization to a table (genomic distance).

#     The normalized value is computed as the log2 of 1 plus the ratio of the
#     value of the bin and some normalization factor. The normalization factor
#     is computed as the median of the values of the bins at a given distance.
#     Raises and error if inter-chromosomal pixels are found.

#     Parameters
#     ----------
#     apply_col : str
#         Column to apply the normalization to.
#     """

#     def __init__(self, *, apply_col: str):
#         self._apply_col = apply_col

#     def process(self, table: "Table") -> "DfChunks":

#         # TODO: maybe add check for inter chromosomal

#         def iter_with_dist(table_obj: "Table"):
#             """Iter chunks with genomic distance. Add inter-chromosomal check."""

#             chunks = table_obj.chunks(annotated=True)
#             bin_size = table_obj.bin_size

#             for chunk in chunks:
#                 yield chunk.with_columns(
#                     ((pl.col("bin2_id") - pl.col("bin1_id")) * bin_size).alias("dist")
#                 )

#         chunks = chunked.chunked_quants(
#             iter_with_dist(table),
#             column=self._apply_col,
#             quants=0.5,
#             split_on=["chrom1", "chrom2"],
#             group_by="dist",
#         )

#         for chunk in chunks:
#             yield chunk.with_columns(
#                 (pl.col(self._apply_col) / pl.col("0.5") + 1).log().alias("norm")
#             )


class NormGenomicDist(NormOperation):
    """Testing genomic mean distance normalization."""

    def __init__(self, *, apply_col: str):
        self._apply_col = apply_col

    def process(self, table: "Table") -> "DfChunks":

        chunks = chunked.chunked_groupby(
            table.chunks(annotated=True), split_on=["chrom1", "chrom2"]
        )

        add_genomic_dist = (pl.col("bin2_id") - pl.col("bin1_id")) * table.bin_size
        for chunk in chunks:
            chunk = (
                chunk.with_columns(add_genomic_dist.alias("dist"))
                .with_columns(
                    pl.col(self._apply_col).mean().over("dist").alias("norm_factor")
                )
                .with_columns(
                    (pl.col(self._apply_col) / pl.col("norm_factor")).alias("norm")
                )
            )
            yield chunk
