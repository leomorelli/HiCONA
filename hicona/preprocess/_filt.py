"""Default classes for pixel table filtering."""

from typing import TYPE_CHECKING

import polars as pl

from hicona._ops import chunked
from hicona.preprocess._abcs import FiltOperation

if TYPE_CHECKING:
    from hicona._core import Table
    from hicona._dtypes import DfChunks


__all__ = [
    "FiltGenomicDist",
    "FiltColumnQuant",
    "FiltColumnValue",
    "FiltInterChroms",
    "FiltSelfLooping",
]


def _within_range(
    column: str,
    lower: int | float | None = None,
    upper: int | float | None = None,
    keep_extrema: bool = True,
) -> pl.Expr | bool:
    """Remove pixels with column value outside of interval [lower, upper]."""

    if keep_extrema:
        above_min = pl.col(column) >= lower if lower is not None else True
        below_max = pl.col(column) <= upper if upper is not None else True
    else:
        above_min = pl.col(column) > lower if lower is not None else True
        below_max = pl.col(column) < upper if upper is not None else True

    return above_min & below_max


class FiltGenomicDist(FiltOperation):
    """Remove pixels whose genomic distance is outside of an interval.

    Genomic distance is calculated as the difference between ``bin1_id`` and
    ``bin2_id`` multiplied by the bin size (in bp). The pixels are removed if
    the distance is outside of the interval (extrema are kept).
    Genomic distance is computed only within the same chromosome.

    Parameters
    ----------
    min_dist : int or None, optional
        Remove pixels whose genomic distance is greater than this value.
        Default is 'None'.
    max_dist : int or None, optional
        Remove pixels whose genomic distance is smaller than this value.
        Default is 'None'.

    """

    def __init__(self, *, min_dist: int | None = None, max_dist: int | None = None):
        if not any([min_dist, max_dist]):
            raise ValueError("At least one distance threshold must be provided.")
        self._min_dist = min_dist
        self._max_dist = max_dist

    def process(self, table: "Table") -> "DfChunks":

        def _get_dist_column(bin_size: int, interchrom: int) -> pl.Expr:
            """Expression to compute genomic distance between bins.

            Genomic distance is calculated as the difference between bin1_id and
            bin2_id multiplied by the bin size (in bp). Pixels on different
            chromosomes are assigned a fixed distance value (which should be in
            the kept interval.)
            """
            exp = (
                pl.when(pl.col("chrom1") != pl.col("chrom2"))
                .then(pl.lit(interchrom))
                .otherwise((pl.col("bin2_id") - pl.col("bin1_id")) * bin_size)
                .alias("dist")
            )
            return exp

        # Set interchromosomal distances to a value which will not be discarded
        inter_value = self._min_dist if self._min_dist is not None else self._max_dist
        assert inter_value is not None

        for chunk in table.chunks(annotated=True):

            chunk = chunk.with_columns(
                _get_dist_column(table.bin_size, inter_value)
            ).filter(_within_range("dist", self._min_dist, self._max_dist, True))

            yield chunk


class FiltColumnQuant(FiltOperation):
    """Remove pixels with column value outside a certain quantile range.

    Filter out pixels where the value of the specified column is outside a
    certain quantile threshold range (extrema are excluded).
    For each threshold, the corresponding value is computed, then all pixels
    with that value are removed. This means that if the bottom 10% pixels
    share the same value, asking to remove the bottom 1% will remove more all
    10%. This is done to avoid arbitrary tie breaks.

    Parameters
    ----------
    apply_col : str
        The column to apply the quantile filter to.
    lower_quant : float or None, optional
        Remove pixels whose column value is not greater than this quantile.
    upper_quant : float or None, optional
        Remove pixels whose column value is not smaller than this quantile.
    chrom_wise : bool, optional
        If True, compute quantiles for each pair of chromosomes separately.
        Default is True.

    """

    def __init__(
        self,
        *,
        apply_col: str,
        lower_quant: float | None = None,
        upper_quant: float | None = None,
        chrom_wise: bool = True,
    ):
        if not any([lower_quant, upper_quant]):
            raise ValueError("At least one quantile threshold must be provided.")
        self._apply_col = apply_col
        self._lower_quant = lower_quant
        self._upper_quant = upper_quant
        self._chrom_wise = chrom_wise

    def process(self, table: "Table") -> "DfChunks":

        lower, upper, col = self._lower_quant, self._upper_quant, self._apply_col

        split_cols = ["chrom1", "chrom2"] if self._chrom_wise else None
        quants = [q for q in (lower, upper) if q is not None]

        # Compute quantiles for each chromosome
        chunks = chunked.chunked_quants(
            table.chunks(annotated=self._chrom_wise),
            column=col,
            quants=quants,
            split_on=split_cols,
        )

        for chunk in chunks:
            yield chunk.filter(
                pl.col(col) > pl.col(str(lower)) if lower is not None else True
            ).filter(pl.col(col) < pl.col(str(upper)) if upper is not None else True)


class FiltColumnValue(FiltOperation):
    """Remove pixels with column value outside a certain interval.

    Filter out pixels where the value of the specified column is outside a
    certain threshold (extrema are kept). Column must be numeric.

    Parameters
    ----------
    apply_col : str
        The column to apply the value filter to.
    lower_value : float or int or None, optional
        Remove pixels whose column value is not greater than this value.
    upper_value : float or int or None, optional
        Remove pixels whose column value is not smaller than this value.

    """

    def __init__(
        self,
        *,
        apply_col: str,
        lower_value: float | int | None = None,
        upper_value: float | int | None = None,
        keep_extrema: bool = True,
    ):
        if not any([lower_value, upper_value]):
            raise ValueError("At least one value threshold must be provided.")
        self._apply_col = apply_col
        self._lower_value = lower_value
        self._upper_value = upper_value
        self._keep_extrema = keep_extrema

    def process(self, table: "Table") -> "DfChunks":

        for chunk in table.chunks():

            yield chunk.filter(
                _within_range(
                    self._apply_col,
                    self._lower_value,
                    self._upper_value,
                    self._keep_extrema,
                )
            )


class FiltInterChroms(FiltOperation):
    """Remove inter-chromosomal pixels from the table.

    Remove pixels whose ``bin1_id`` and ``bin2_id`` are on different chromosomes.

    """

    def process(self, table: "Table") -> "DfChunks":
        for chunk in table.chunks(annotated=True):
            yield chunk.filter(pl.col("chrom1") == pl.col("chrom2"))


class FiltSelfLooping(FiltOperation):
    """Remove self-looping pixels from the table.

    Remove pixels whose ``bin1_id`` and ``bin2_id`` are the same.

    """

    def process(self, table: "Table") -> "DfChunks":
        for chunk in table.chunks():
            yield chunk.filter(pl.col("bin1_id") != pl.col("bin2_id"))
