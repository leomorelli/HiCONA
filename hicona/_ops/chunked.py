"""Functions which modify and return iterables of pixel table chunks."""

import pathlib

import polars as pl

from hicona._dtypes import DfChunks, T


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

    # TODO: Make stat choices modular to fetch only needed stats

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


def get_degree_ranking(
    chunks: DfChunks,
    degrees: pl.DataFrame,
    id_breaks: list[tuple[int, int]],
    path: pathlib.Path,
) -> pathlib.Path:
    """Compute the neighbor degree rankings for each node.

    The neighbor degree ranking can be explained as follows: given a node,
    all neighbors of that node are assigned a rank which is equal to the
    number of neighbors of that node which have a higher degree then the
    considered neighbor (plus one).

    Since the method allows for ties, rather than returning the rank of
    each neighbor of each node, for each node it returns the rank of
    each neighbor degree.

    Since the ranking table can be as large as the original pixel table,
    the computation is split into chunks to avoid memory overload and
    the results are saved to parquet files.
    """

    def create_parquet_folders(
        tmp_path: pathlib.Path, id_breaks: list[tuple[int, int]]
    ) -> pathlib.Path:
        """Create the folders in which to create the temporary parquets."""

        for lower, upper in id_breaks:
            (tmp_path / f"{lower}-{upper}").mkdir(parents=True)

        return tmp_path

    def create_chunk_parquets(
        chunks: DfChunks,
        degrees: pl.DataFrame,
        id_breaks: list[tuple[int, int]],
        tmp_path: pathlib.Path,
    ):
        """Compute the rankings for each node and save to split parquets.

        The process is a bit convoluted but it is necessary to avoid memory
        overload since, at worst, the ranking table can be as large as the
        original pixel table.
        """

        # For each pixel chunk, compute how many times each node is
        # connected to a node of a certain degree. (Repeat on both columns).
        # Split the results in folders according to the node id.
        for i, chunk in enumerate(chunks):

            bin_cols = ["bin1_id", "bin2_id"]
            parts: list[pl.DataFrame] = []
            for step in [1, -1]:
                group_col, count_col = bin_cols[::step]

                parts.append(
                    chunk.join(degrees, left_on=group_col, right_on="bin_id")
                    .group_by([count_col, "degree"])
                    .count()
                    .select(pl.col(count_col).alias("bin_id"), "degree", "count")
                )

            chunk = pl.concat(parts)

            for lower, upper in id_breaks:
                chunk.filter(
                    (pl.col("bin_id") >= lower) & (pl.col("bin_id") < upper)
                ).write_parquet(tmp_path / f"{lower}-{upper}" / f"{i}.parquet")

        # NOTE: Somewhere below here, there is a step which makes memory usage
        # spike in a non-linear way when increasing number of nodes per chunk.

        # For each folder, e.i. interval of node ids, aggregate the counts and
        # compute the rankings for each node.
        for lower, upper in id_breaks:
            break_fold = tmp_path / f"{lower}-{upper}"

            # Collect all files within the folder and aggregate the counts
            # (a node can be connected to a degree n node in multiple chunks)
            parquet_df = (
                pl.scan_parquet(break_fold / "*")
                .group_by(["bin_id", "degree"])
                .agg(pl.sum("count"))
                .sort(by=["bin_id", "degree"], descending=[False, True])
                .collect()
            )

            # For each node, convert the neighbor degrees counts to a ranking,
            # starting from 1. Save the results to a new parquet.
            (
                parquet_df.with_columns(
                    parquet_df.group_by("bin_id", maintain_order=True)
                    .agg(
                        pl.col("count")
                        .shift(1)
                        .cum_sum()
                        .fill_null(0)
                        .add(1)
                        .alias("rank")
                    )
                    .explode(pl.col("rank"))
                )
                .drop("count")
                .write_parquet(str(break_fold) + ".parquet")
            )

    tmp_path = create_parquet_folders(path, id_breaks)
    print("Starting node ranking computation...")
    create_chunk_parquets(chunks, degrees, id_breaks, tmp_path)
    print("Node ranking computation finished.")

    return tmp_path


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
                [c.with_columns(*expressions) for _, c in interval.group_by(group_by)]
            ).sort(["bin1_id", "bin2_id"])
        else:
            yield interval.with_columns(*expressions)
