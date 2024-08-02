"""Functions which modify and return iterables of pixel table chunks."""

import polars as pl

from hicona._dtypes import DfChunks, T

DEFAULT_COL = "bin1_id"


def chunked_groupby(
    iterator: DfChunks,
    split_on: list[str],
) -> DfChunks:
    """Groupby operation on an Iterable of chunks.

    Given an iterable of chunks (generally of fixed size), return a new
    iterable of chunks where the chunks are the result of a groupby operation
    spanning across the chunks. It is assumed that the chunks are collectively
    sorted by the columns in `split_on`.
    """

    pixels: pl.DataFrame = pl.DataFrame()

    for new_pixels in iterator:
        pixels = pl.concat([pixels, new_pixels])

        *chunks, (_, pixels) = pixels.group_by(split_on, maintain_order=True)
        for chunk in chunks:
            yield chunk[1]

    for chunk in pixels.group_by(split_on, maintain_order=True):
        yield chunk[1]


def get_node_stats(chunks: DfChunks, weight_col: str) -> pl.DataFrame:
    """Compute sum of weights and degree for each node/bin."""

    stats = []

    for chunk in chunks:
        for bin_col in ["bin1_id", "bin2_id"]:
            # Compute metrics on chunk
            grouped = (
                chunk.select(bin_col, weight_col)
                .group_by(bin_col)
                .agg(
                    [
                        pl.col(weight_col).sum().alias("weight"),
                        pl.count(bin_col).alias("degree"),
                    ]
                )
                .rename({bin_col: "bin_id"})
            )

            stats.append(grouped)

    return pl.concat(stats).group_by("bin_id").sum()


# def get_node_count_freq(chunks: DfChunks, weight_col: str) -> pd.DataFrame:
#     """Return sorted edge weights for each node in the network.

#     Returns a dictionary where the keys are the node IDs and the values are
#     lists of the edge weights connected to that node. The lists are sorted in
#     descending order.
#     """

#     node_weights = None
#     for chunk in chunks:
#         bin_cols = ["bin1_id", "bin2_id"]
#         for step in [1, -1]:
#             group_col, count_col = bin_cols[::step]
#             counts = chunk.rename(columns={group_col: "bin_id", count_col: "freq"})
#             counts = counts.groupby(["bin_id", weight_col]).count()
#             counts = counts[["freq"]]

#             if node_weights is None:
#                 node_weights = counts
#             else:
#                 node_weights = node_weights.add(counts, fill_value=0)

#     assert node_weights is not None

#     node_weights["freq"] = node_weights["freq"].astype(int)
#     node_weights.reset_index(inplace=True)
#     node_weights.sort_values(by=["bin_id", weight_col], inplace=True)

#     def calc_ranks(df):
#         df["rank"] = df["freq"].cumsum().shift(fill_value=0) + 1
#         return df

#     node_weights = node_weights.groupby("bin_id").apply(calc_ranks)
#     node_weights.reset_index(drop=True, inplace=True)

#     node_weights["degree"] = node_weights.groupby(["bin_id"])["freq"].transform("sum")
#     node_weights.drop(columns=["freq"], inplace=True)

#     assert isinstance(node_weights, pd.DataFrame)

#     return node_weights


def chunked_quants(
    iterator: DfChunks,
    column: str,
    quants: float | list[float],
    split_on: None | str | list[str] = None,
    group_by: None | str | list[str] = None,
) -> DfChunks:
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

    intervals = chunked_groupby(iterator, split_on) if split_on else iterator
    expressions = [pl.col(column).quantile(q, "linear").alias(str(q)) for q in quantile]

    for interval in intervals:

        if group_by:
            yield pl.concat(
                [c.with_columns(*expressions) for _, c in interval.groupby(group_by)]
            ).sort(["bin1_id", "bin2_id"])
        else:
            yield interval.with_columns(*expressions)
