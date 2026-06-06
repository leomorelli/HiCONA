"""Provided count normalization functions."""

import math
from typing import Callable

import polars as pl

PixNormFunc = Callable[[pl.LazyFrame, str], pl.LazyFrame]


def norm_arctan_mean(lf: pl.LazyFrame, column: str) -> pl.LazyFrame:
    """Apply arctan normalization: ``arctan(count / mean_count) / (pi/2)``.

    Resulting values lie in ``[0, 1]`` for non-negative counts.
    """

    # No need to filter out zeros as "count" column already contains non-zeros
    ave_non_zero_count = lf.select("count").mean().collect().item()

    return lf.with_columns(
        pl.col("count")
        .truediv(ave_non_zero_count)
        .arctan()
        .truediv(math.pi / 2)
        .alias(column)
    )


def norm_log(lf: pl.LazyFrame, column: str) -> pl.LazyFrame:
    """Apply natural log transform with a ``+1`` pseudo-count: ``ln(count + 1)``."""
    return lf.with_columns(pl.col("count").add(1).log().alias(column))


# Collection of normalization function
norm_functions: dict[str, PixNormFunc] = {
    "arctan_mean": norm_arctan_mean,
    "log": norm_log,
}
