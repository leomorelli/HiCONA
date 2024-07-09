"""Default classes for pixel table normalization."""

from typing import TYPE_CHECKING as _TYPE_CHECKING

import numpy as np

from hicona._ops import chunked
from hicona.preprocess._abcs import NormOperation

if _TYPE_CHECKING:
    from hicona._core import Table
    from hicona._dtypes import PdChunks


__all__ = ["NormBinwise", "NormGenomicDist"]


# class NormNone(NormOperation):
#     """Apply no normalization to a table.

#     Do not apply any normalization to the table, just return the original
#     count values as the normalized values. This is used to copy the raw
#     values to the normalized column when no normalization was applied.
#     """

#     def run(self, table: "Table") -> "PdChunks":
#         for chunk in table.chunks():
#             chunk["norm"] = chunk["count"]
#             yield chunk[["bin1_id", "bin2_id", "count", "norm"]]


class NormBinwise(NormOperation):
    """Apply a binwise normalization to a table.

    The normalization factors to use for normalization must be preemtively
    added to the cooler as a column in the bin table.

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
    This function mimics matrix balancing normalization found in cooler.
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

    def run(self, table: "Table") -> "PdChunks":

        col1, col2 = f"{self._ann_name}1", f"{self._ann_name}2"

        for chunk in table.chunks(annotated=True):
            if self._divisive:
                chunk[col1] = 1 / chunk[col1]
                chunk[col2] = 1 / chunk[col2]
            chunk["norm"] = chunk[self._apply_col] * chunk[col1] * chunk[col2]

            if self._drop_nas:
                chunk.dropna(inplace=True)

            yield chunk[["bin1_id", "bin2_id", "count", "norm"]]


class NormGenomicDist(NormOperation):
    """Apply default HiCONA normalization to a table (genomic distance).

    The normalized value is computed as the log2 of 1 plus the ratio of the
    value of the bin and some normalization factor. The normalization factor
    is computed as the median of the values of the bins at a given distance.
    Raises and error if inter-chromosomal pixels are found.

    Parameters
    ----------
    apply_col : str
        Column to apply the normalization to.
    """

    def __init__(self, *, apply_col: str):
        self._apply_col = apply_col

    def run(self, table: "Table") -> "PdChunks":

        def distance_iter(table_obj):
            """Iter chunks with genomic distance. Add inter-chromosomal check."""

            chunks = table_obj.chunks(annotated=True)
            bin_size = table_obj.bin_size

            for chunk in chunked.add_gen_dist(chunks, bin_size):

                # Check that inter-chromosomal pixels where removed
                if any(chunk["chrom1"] != chunk["chrom2"]):
                    raise ValueError("Inter-chromosomal pixels must be removed.")

                yield chunk

        norm_curve = chunked.chunked_quants(
            distance_iter(table),
            column=self._apply_col,
            quants=0.5,
            split_on=["chrom1", "chrom2"],
            group_by="dist",
        )
        norm_curve.rename(columns={self._apply_col: "dist_norm"}, inplace=True)

        for chunk in distance_iter(table):
            chunk = chunk.merge(norm_curve, how="left", on=["chrom1", "dist"])
            chunk["norm"] = np.log2(chunk[self._apply_col] / chunk["dist_norm"] + 1)

            yield chunk[["bin1_id", "bin2_id", "count", "norm"]]
