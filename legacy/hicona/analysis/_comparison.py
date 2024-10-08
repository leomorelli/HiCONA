"""Placeholder."""

from typing import Union
from itertools import combinations

import numpy as np
import polars as pl

from hicona._ops import uris
from hicona._core import PixelTable
from hicona.analysis import _plotting
from hicona.preprocess import Flow


def compare_tables(
    table_a: "PixelTable",
    table_b: Union["PixelTable", None],
    values: str = "count",
    region: str | None = None,
    *,
    log_scale: bool = False,
    binary: bool = False,
    match_coverage: bool = False,
    **kwargs,
):
    """Placeholder"""

    def prepare_table(
        table: "PixelTable",
        lower: int,
        upper: int,
        value_col: str,
    ) -> pl.DataFrame:
        """Prepare the table for comparison."""

        return (
            table.dataframe()
            .filter(
                pl.col("bin1_id").ge(lower),
                pl.col("bin1_id").le(upper),
                pl.col("bin2_id").ge(lower),
                pl.col("bin2_id").le(upper),
            )
            .with_columns(
                pl.col("bin1_id") - lower,
                pl.col("bin2_id") - lower,
            )
            .select(["bin1_id", "bin2_id", value_col])
            .collect()
        )

    # PixelTable -> other table or self filtered
    # none -> full pixel table from file

    # The full pixel table only has bin1_id, bin2_id, and count columns, so a comparison
    # with any other column is not possible.
    if not table_b and values != "count":
        raise ValueError("Only counts are available when comparing to full table.")

    # If no table is provided, fetch the full table from the same file
    if not table_b:
        pixels_uris = uris.Uris(table_a.path.store, table_a.path.root, "pixels")
        table_b = PixelTable.from_cooler(pixels_uris, table_a.chunk_size)

    if match_coverage:
        raise NotImplementedError("Match coverage is not yet implemented.")

    plot_kwargs = {"values_col": values, "log_scale": log_scale, "binary": binary}

    if region:
        lower, upper = table_a.bins.get_region_bounds(region)
        fig, plot = _plotting.plot_table_comparison(
            prepare_table(table_a, lower, upper, values),
            prepare_table(table_b, lower, upper, values),
            **plot_kwargs,
            **kwargs,
        )
        fig.suptitle(region)
    else:
        fig, plot = _plotting.plot_table_comparison(
            table_a.dataframe().collect(),
            table_b.dataframe().collect(),
            **plot_kwargs,
            **kwargs,
        )

    return fig, plot


def _single_pair_jaccard(
    table_a: PixelTable,
    table_b: PixelTable,
    chunk_size: int = 10_000_000,
) -> float:
    """Placeholder"""

    def slicing_row(df_a: pl.DataFrame, df_b: pl.DataFrame) -> tuple[int, int]:
        """Given two dataframes, take the last row of each, then return bin1_id
        and bin2_id of the row which comes first in COO format."""

        bin1_a, bin2_a = df_a.select(pl.last("bin1_id", "bin2_id"))
        bin1_b, bin2_b = df_b.select(pl.last("bin1_id", "bin2_id"))

        bin1_a, bin2_a = bin1_a[0], bin2_a[0]
        bin1_b, bin2_b = bin1_b[0], bin2_b[0]

        if not bin1_a:  # Exhausted table a
            return bin2_a, bin2_b
        if not bin1_b:  # Exhausted table b
            return bin1_a, bin1_b
        if bin1_a < bin1_b:  # Table a bin1 exhausts first
            return bin1_a, bin2_a
        if bin1_a == bin1_b:  # Bin 1 exhaust at the same time, get bin2 exhaust
            return bin1_a, min(bin2_a, bin2_b)
        return bin1_b, bin2_b  # Table b bin1 exhaust first

    def get_sliced_parts(
        df: pl.DataFrame, vals: tuple[int, int]
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Given a dataframe and slicing values, return the sliced parts."""
        # If this code ever reaches production, this should be optimzed
        # currently requires two passes over the data (keep and discard)
        # one could probably suffice using some form of groupby

        bin1_cond = pl.col("bin1_id") < vals[0]
        bin2_cond = (pl.col("bin1_id") == vals[0]) & (pl.col("bin2_id").le(vals[1]))
        filt_cond = bin1_cond | bin2_cond

        return df.filter(filt_cond), df.filter(~filt_cond)

    remainder_a: pl.DataFrame = pl.DataFrame()
    remainder_b: pl.DataFrame = pl.DataFrame()

    # Accumulators
    inter: int = 0
    union: int = 0

    max_table_size: int = max(table_a.get_size(), table_b.get_size())
    for i in range(0, max_table_size, chunk_size):
        # Get the new chunks and add any remainder from the previous chunk
        chunk_a = table_a.dataframe().slice(i, i + chunk_size).collect()
        chunk_b = table_b.dataframe().slice(i, i + chunk_size).collect()
        chunk_a = pl.concat([remainder_a, chunk_a])
        chunk_b = pl.concat([remainder_b, chunk_b])

        # Slice the shared interval
        slicing_vals = slicing_row(chunk_a, chunk_b)
        chunk_a, remainder_a = get_sliced_parts(chunk_a, slicing_vals)
        chunk_b, remainder_b = get_sliced_parts(chunk_b, slicing_vals)

        # Compute the amount of intersecting and union lines in the interval
        inter += chunk_a.join(chunk_b, on=["bin1_id", "bin2_id"], how="inner").height
        union += chunk_a.join(chunk_b, on=["bin1_id", "bin2_id"], how="outer").height

    assert not (
        remainder_a.height != 0 and remainder_b.height != 0
    ), "It should not be possibile to have remainder for both tables."

    # Add any remaining non intersecting lines
    union += remainder_a.height + remainder_b.height

    min_val = min(
        table_a.dataframe().min().collect()["bin1_id"][0],
        table_b.dataframe().min().collect()["bin1_id"][0],
    )
    max_val = max(
        table_a.dataframe().max().collect()["bin2_id"][0],
        table_b.dataframe().max().collect()["bin2_id"][0],
    )
    tot_size = int((max_val - min_val))
    # print(tot_size)
    # print(inter)
    # print(union)
    # jaccard = 1 - ((union - inter) / (tot_size**2))
    # print(jaccard)

    # return jaccard
    return inter / union


def _pairwise_jaccard(
    tables: list[PixelTable],
    names: list[str],
    flow: Flow,
) -> pl.DataFrame:
    """Placeholder"""

    new_tables = [table.apply(flow) for table in tables]
    new_rows = pl.DataFrame(
        [
            {
                "Table A": n1,
                "Table B": n2,
                "Jaccard": _single_pair_jaccard(t1, t2),
            }
            for (t1, t2), (n1, n2) in zip(
                combinations(new_tables, 2), combinations(names, 2)
            )
        ]
    )

    return new_rows.with_columns(pl.lit(flow.name).alias("Flow"))


def pairwise_jaccard(
    tables: list[PixelTable],
    names: list[str] | None = None,
    flows: list["Flow"] | None = None,
) -> pl.DataFrame:
    """Placeholder"""

    names = names if names else [f"Table {i}" for i in range(len(tables))]
    flows = [Flow(name="Raw")] + (flows if flows else [])

    assert len(names) == len(tables), "Length of names must match tables."
    assert len(tables) > 1, "At least two tables are required for comparison."

    return pl.concat([_pairwise_jaccard(tables, names, flow) for flow in flows])
