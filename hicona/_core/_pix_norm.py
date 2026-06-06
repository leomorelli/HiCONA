"""Provided count normalization functions."""

import math
from typing import Callable

import polars as pl

PixNormFunc = Callable[[pl.LazyFrame, str], pl.LazyFrame]


def norm_arctan_mean(lf: pl.LazyFrame, column: str) -> pl.LazyFrame:
    """Apply arctan normalization to counts divided by average count.

    y = arctan(x/ave_count) / (pi/2)

    ave_count = average of non-zero pixels in the table

    y in [-1, +1] if x in (-inf, +inf)
    y in [ 0, +1] if x in [   0, +inf)
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
    """Log transform (with +1 pseudo-count)."""
    return lf.with_columns(pl.col("count").add(1).log().alias(column))


# Collection of normalization function
norm_functions: dict[str, PixNormFunc] = {
    "arctan_mean": norm_arctan_mean,
    "log": norm_log,
}
