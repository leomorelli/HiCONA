"""Functions which modify and return iterables of pixel table chunks."""

import pandas as pd

from hicona._dtypes import PdChunks, T


DEFAULT_COL = "bin1_id"


def change_breaks(
    iterator: PdChunks,
    split_on: list,
    to_keep_cols: list[str],
) -> PdChunks:
    """Return a generator of intervals based on the split_on columns.

    Rather then using the iterator with regular chunks, divide the chunks
    based on where the value of one of the split columns changes. Basically
    it is a groupby operation, but faster and more memory efficient when the
    groups are ordered. The main use it to split for pairs of chromosomes.
    """

    curr_interval: list[pd.DataFrame] = []
    prev_values: list = [None] * len(split_on)

    for chunk in iterator:

        split_cols = chunk[split_on]
        value_cols = chunk[to_keep_cols + split_on]

        breaks = pd.Series([False] * len(chunk))
        for col, val in zip(split_cols.columns, prev_values):
            val = val or split_cols[col].iloc[0]
            col = split_cols[col]
            breaks = breaks | (col != col.shift(fill_value=val))
        breaks = list(breaks[breaks].index)

        for b in breaks:
            curr_interval.append(value_cols.loc[: b - 1])
            value_cols = value_cols.loc[b:]

            interval = pd.concat(curr_interval)
            curr_interval = []

            yield interval

        curr_interval.append(value_cols)

    interval = pd.concat(curr_interval)
    yield interval


def add_bin_diff(iterator: PdChunks) -> PdChunks:
    """Add a column with the difference between bin2 and bin1."""

    for chunk in iterator:
        chunk["diff"] = chunk["bin2_id"] - chunk["bin1_id"]
        yield chunk


def add_gen_dist(iterator: PdChunks, bin_size: int) -> PdChunks:
    """Add a column with the genomic distance between bin2 and bin1."""

    for chunk in iterator:
        chunk["dist"] = (chunk["bin2_id"] - chunk["bin1_id"]) * bin_size
        yield chunk


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
