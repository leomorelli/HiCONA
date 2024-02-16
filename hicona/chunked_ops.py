"""Utility functions for DataFrame iterators yielding a single result.

Set of utility functions which can be applied on a iterator of pandas
DataFrames yielding a single aggregated result. This way it should be
possible to apply these functions to bigger than memory dataframes.
"""

import pandas as pd

DEFAULT_COL = "bin1_id"


def get_col_quantiles(iterator, colname, quants):
    """Placeholder"""

    quant = [quant] if isinstance(quant, float) else quant

    curve = pd.Series()
    for chunk in iterator:
        vals = chunk.groupby(colname)[DEFAULT_COL].count()
        curve = curve.combine(vals, lambda x, y: x + y, fill_value=0)

    tot_items = curve.sum()
    cum_curve = curve.cumsum()

    values = []
    for quant in quants:
        threshold = tot_items * quant
        values.append(cum_curve[cumulative > threshold].index[0])

    return values


def get_node_stats(iterator, weight_col):
    """Compute sum of weights and degree for each node/bin."""

    weights = pd.Series()
    degrees = pd.Series()

    for chunk in iterator:
        for bin_col in ["bin1_id", "bin2_id"]:
            # Compute metrics on chunk
            grouped = chunk.groupby(bin_col)
            chunk_weights = grouped.sum()[weight_col]
            chunk_degrees = grouped.count()[weight_col]

            # Increase counters
            weights = weights.add(chunk_weights, fill_value=0)
            degrees = degrees.add(chunk_degrees, fill_value=0)

    return pd.DataFrame({"weight": weights, "degree": degrees})
