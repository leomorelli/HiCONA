"""Utility functions for DataFrame chunkss yielding a single result.

Set of utility functions which can be applied on a chunks of pandas
DataFrames yielding a single aggregated result. This way it should be
possible to apply these functions to bigger than memory dataframes.
"""

import pandas as pd

from .iterable_ops import change_breaks
from .dtypes import PdChunks, T

DEFAULT_COL = "bin1_id"


def get_node_stats(chunks: PdChunks, weight_col: str) -> pd.DataFrame:
    """Compute sum of weights and degree for each node/bin."""

    weights = pd.Series()
    degrees = pd.Series()

    for chunk in chunks:
        for bin_col in ["bin1_id", "bin2_id"]:
            # Compute metrics on chunk
            grouped = chunk.groupby(bin_col)
            chunk_weights = grouped.sum()[weight_col]
            chunk_degrees = grouped.count()[weight_col]

            # Increase counters
            weights = weights.add(chunk_weights, fill_value=0)
            degrees = degrees.add(chunk_degrees, fill_value=0)

    return pd.DataFrame({"weight": weights, "degree": degrees})


def chunked_quants(
    iterator: PdChunks,
    column: str,
    quants: float | list[float],
    split_on: None | str | list[str] = None,
    group_by: None | str | list[str] = None,
) -> pd.DataFrame:
    """Compute quantiles on an Iterable of chunks.

    Compute quantiles for a table provided as an iterator of chunks. The
    quantiles can be computed on an entire column or on subsets of it defined
    by groupby operations. Since groupby is a costly operation, the parameter
    `split_on` can be used as a faster alternative when the groups are
    ordered (even if they are split into chunks).

    NOTE: `group_by` without `split_on` can be very memory intensive,
    especially when applied to a full genome pixel table.
    """

    def to_list(item: None | T | list[T]) -> list[T]:
        """Convert to list of strings if not already"""
        item = item or []
        return item if isinstance(item, list) else [item]

    quantile = to_list(quants)
    split_on = to_list(split_on)
    group_by = to_list(group_by)

    results: list[pd.DataFrame] = []
    intervals = change_breaks(iterator, split_on, [column] + group_by)
    for interval in intervals:

        # NOTE: np.ndarray does not implement __len__ so it is not array-like
        # according to  pandas. Using list even though typing raises an alert.
        if group_by:
            qvals = interval.groupby(group_by)[column].quantile(quantile)  # type: ignore
            col_fix = {"level_1": "quant"}
        else:
            qvals = interval[column].quantile(quantile)
            col_fix = {"index": "quant"}

        quants_df = qvals.reset_index()
        quants_df.rename(columns=col_fix, inplace=True)

        if split_on:
            for col in split_on:
                quants_df[col] = interval[col].iloc[0]

        results.append(quants_df)

    return pd.concat(results)
