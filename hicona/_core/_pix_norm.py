"""Provided count normalization functions."""

import math
from typing import Callable

import polars as pl

PixNormFunc = Callable[[pl.LazyFrame, str], pl.LazyFrame]


def norm_arctan(lf: pl.LazyFrame, column: str) -> pl.LazyFrame:
    """Apply arctan normalization to counts.

    This normalization is particularly amenable for clustering since
    it brings values in the range [0, 1].

    """

    ave_counts = (
        lf.with_columns(pl.col("bin2_id").sub("bin1_id").alias("dist"))
        .select(("dist", "count"))
        .group_by("dist")
        .mean()
        .rename({"count": "_norm_factor"})
    )

    return (
        lf.with_columns(pl.col("bin2_id").sub("bin1_id").alias("dist"))
        .join(ave_counts, on="dist", how="left")
        .with_columns(
            pl.col("count")
            .truediv("_norm_factor")
            .arctan()
            .truediv(math.pi)  # Scale in [-0.5, 0.5] range
            .add(0.5)  # Move to [0, 1] range
            .alias(column)
        )
        .drop("_norm_factor")
    )


def norm_log(lf: pl.LazyFrame, column: str) -> pl.LazyFrame:
    """Log transform (with +1 pseudo-count)."""
    return lf.with_columns(pl.col("count").add(1).log().alias(column))


# Collection of normalization function
norm_functions: dict[str, PixNormFunc] = {
    "arctan": norm_arctan,
    "log": norm_log,
}
