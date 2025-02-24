"""Modifiers for table data fetching.

Functions which take as input a in iterable of polars chunks and return a
modified version of it. In general, these functions should be able to be
made into partial functions, to be able to be applied in a chain.

"""

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from .._utils.df_dtypes import PlChunks


__all__ = ("annotate_pixels", "balance_pixels", "subset_region")


def subset_region(chunks: "PlChunks", *, extent: tuple[int, int]) -> "PlChunks":
    """Subset the chunks to a genomic region.

    Parameters
    ----------
    chunks : PlChunks
        Iterable of polars chunks.
    extent : tuple[int, int]
        Genomic region boundaried (as bin ids).

    Returns
    -------
    PlChunks
        Subsetted chunks.

    """

    lower, upper = extent
    pix_filt: pl.Expr = (
        (pl.col("bin1_id") >= lower)
        & (pl.col("bin1_id") < upper)
        & (pl.col("bin2_id") >= lower)
        & (pl.col("bin2_id") < upper)
    )

    for chunk in chunks:
        yield chunk.filter(pix_filt)


def annotate_pixels(chunks: "PlChunks", *, bins_df: pl.DataFrame) -> "PlChunks":
    """Annotate the pixels with bin information.

    Parameters
    ----------
    chunks : PlChunks
        Iterable of polars chunks.
    bins_df : pl.DataFrame
        DataFrame containing the bin information.

    Returns
    -------
    PlChunks
        Annotated chunks.

    """

    # TODO: decide whether to subsect columns (especially weight)
    for chunk in chunks:
        yield (
            chunk.join(bins_df, how="left", left_on="bin1_id", right_on="bin_id")
            .join(bins_df, how="left", left_on="bin2_id", right_on="bin_id", suffix="2")
            .rename({c: c + "1" for c in bins_df.columns if c != "bin_id"})
        )


def balance_pixels(
    chunks: "PlChunks",
    *,
    bins_df: pl.DataFrame,
    drop_nulls: bool = True,
) -> "PlChunks":
    """Balance the pixel counts by the bin weights.

    It is assumed that the bin weights are stored in the 'weight' column.
    Also assumed that the weights are multiplicative, as in cooler.

    Parameters
    ----------
    chunks : PlChunks
        Iterable of polars chunks.
    bins_df : pl.DataFrame
        DataFrame containing the bin information.
    drop_nulls : bool
        Whether to remove pixels whose count column became null during balancing.

    Returns
    -------
    PlChunks
        Balanced pixel counts.

    """

    if "weight" not in bins_df.columns:
        raise ValueError("No 'weight' column found. Run `cooler balance` first.")

    weights: pl.DataFrame = bins_df.select(["bin_id", "weight"])
    for chunk in chunks:
        yield (
            chunk.join(weights, how="left", left_on="bin1_id", right_on="bin_id")
            .with_columns((pl.col("count") * pl.col("weight")).alias("count"))
            .join(weights, how="left", left_on="bin2_id", right_on="bin_id", suffix="2")
            .with_columns((pl.col("count") * pl.col("weight2")).alias("count"))
            .drop("weight", "weight2")
            .filter(pl.col("count").is_not_null() if drop_nulls else pl.lit(True))
        )
