"""Functions which modify and return iterables of pixel table chunks."""

import pandas as pd

from ._dtypes import PdChunks


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
