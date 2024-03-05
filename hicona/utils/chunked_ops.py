"""Utility functions for DataFrame chunkss yielding a single result.

Set of utility functions which can be applied on a chunks of pandas
DataFrames yielding a single aggregated result. This way it should be
possible to apply these functions to bigger than memory dataframes.
"""

from typing import Iterable

import pandas as pd
import numpy as np


DEFAULT_COL = "bin1_id"


def col_quants(
    chunks: Iterable[pd.DataFrame],
    colname: str,
    quants: list[float] | float,
) -> list[float]:
    """Placeholder"""

    # NOTE: workaround for assumed bug in type hinting
    qvals: list[float] = [quants] if isinstance(quants, float) else quants  # type: ignore

    curve = pd.Series()
    for chunk in chunks:
        vals = chunk.groupby(colname)[DEFAULT_COL].count()
        curve = curve.combine(vals, lambda x, y: x + y, fill_value=0)

    tot_items = curve.sum()
    cum_curve = curve.cumsum()

    values = []
    for quant in qvals:
        threshold = tot_items * quant
        values.append(cum_curve[cum_curve > threshold].index[0])

    return values


def get_node_stats(
    chunks: Iterable[pd.DataFrame],
    weight_col: str,
) -> pd.DataFrame:
    """Compute sum of weights and degree for each node/bin."""

    weights = pd.Series()
    degrees = pd.Series()

    for chunk in chunks:
        print("COMPUTING NODE STATS")
        print(chunk)
        for bin_col in ["bin1_id", "bin2_id"]:
            # Compute metrics on chunk
            grouped = chunk.groupby(bin_col)
            chunk_weights = grouped.sum()[weight_col]
            chunk_degrees = grouped.count()[weight_col]

            # Increase counters
            weights = weights.add(chunk_weights, fill_value=0)
            degrees = degrees.add(chunk_degrees, fill_value=0)

    return pd.DataFrame({"weight": weights, "degree": degrees})


def groupwise_median(
    chunks: Iterable[pd.DataFrame],
    group_cols: list[str],
    value_col: str,
) -> pd.Series:
    """Compute the median of a value column for each group."""

    curve = pd.Series()

    for chunk in chunks:
        parts = chunk.groupby(group_cols, observed=True)[value_col].apply(list)
        curve = curve.combine(parts, lambda x, y: x + y, fill_value=[])  # type: ignore
        # NOTE: [] is not an explicitely supported value, but it works
        # TODO: Maybe change to a better solution (also lists might be slow)

    curve = curve.rename_axis(index=group_cols)

    return curve.apply(np.median).rename(value_col)


def groupwise_quants(
    chunks: Iterable[pd.DataFrame],
    group_cols: list[str],
    value_col: str,
    quants: list[float] | float,
) -> pd.Series:
    """Compute the quantiles of a value column for each group."""

    curve = pd.Series()

    for chunk in chunks:
        parts = chunk.groupby(group_cols, observed=True)[value_col].apply(list)
        curve = curve.combine(parts, lambda x, y: x + y, fill_value=[])  # type: ignore
        # NOTE: [] is not an explicitely supported value, but it works
        # TODO: Maybe change to a better solution (also lists might be slow)

    curve = curve.rename_axis(index=group_cols)

    return curve.apply(np.quantile, args=(quants,)).rename(value_col)
