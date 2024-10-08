"""General utility table operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from .._utils.chunked_ops import convert, row_filter, to_iterable, format_stream


if TYPE_CHECKING:
    from .._utils.df_dtypes import DataFrame, DfChunks, DfStream, DfDtype, Bool


__all__ = ["find_extent", "subset_pixel_region", "subset_bin_region"]


def find_extent(bins: "DfStream", region: str) -> tuple[int, int]:
    """Find the lower and upper bin ids for a given genomic region.

    Analogous to cooler.Cooler.extent method, but not tied to a Cooler object.
    """

    def bins_filter(region: str, res: int) -> pl.Expr:
        """Filter to apply on the bins to find boundary ids."""
        chrom, interval = region.split(":")
        bins_expr = pl.col("chrom") == chrom

        if interval:
            start, end = map(int, interval.split("-"))
            start = (start // res) * res
            end = (end // res + 1 if end % res else end // res) * res
            bins_expr &= (pl.col("start") >= start) & (pl.col("end") <= end)

        return bins_expr

    # Containers for the results
    lower_id: int | None = None
    upper_id: int | None = None
    bin_expr: pl.Expr | None = None

    for bins_chunk in convert(bins, "polars"):

        # NOTE: Filtering expression is defined during iteration because it is not
        #       possible to peek without being sure not to consume the chunk.
        if not bin_expr:
            row: dict[str, str | int] = bins_chunk.row(0, named=True)
            res: int = int(row["end"]) - int(row["start"])
            bin_expr = bins_filter(region, res)

        # Filter the chunk and act according to how many rows are left
        # This flowchart assumes that the bins are sorted and contiguous.
        # TODO: maybe enforce that the bins are sorted and contiguous
        # 0 rows:
        #    - If the upper bound is set, you overshot the region and can stop.
        #    - If the lower bound is not set, you are before the region.
        # 0 < rows < chunk_size:
        #    - You found the top of the region, set the upper bound.
        # chunk_size rows:
        #    - It might be the top of the region, but cannot be sure. Continue.
        bins_chunk = bins_chunk.filter(bin_expr)

        if bins_chunk.height == 0:
            if upper_id is not None:
                break
            continue

        if lower_id is None:
            lower_id = int(bins_chunk["bin_id"].min())  # type: ignore
        upper_id = int(bins_chunk["bin_id"].max())  # type: ignore
        # TODO: maybe enforce type check for column dtypes

    assert lower_id is not None and upper_id is not None
    return lower_id, upper_id


def subset_pixel_region(
    pixels: "DataFrame" | "DfStream",
    *,
    region: str,
    ref_bins: "DataFrame" | "DfStream",
    df_dtype: DfDtype = "polars",
    as_chunks: Bool = True,
    chunk_size: int = 10_000_000,
) -> "DataFrame" | "DfChunks":
    """Return a subset of pixels within a genomic region."""

    def pixels_filter(lower_id: int, upper_id: int) -> pl.Expr:
        """Filter to apply on the pixels to find the subset."""
        bin1_bounds = (pl.col("bin1_id") >= lower_id) & (pl.col("bin1_id") < upper_id)
        bin2_bounds = (pl.col("bin2_id") >= lower_id) & (pl.col("bin2_id") < upper_id)
        return bin1_bounds & bin2_bounds

    pixels, ref_bins = to_iterable(pixels, ref_bins)
    lower_id, upper_id = find_extent(ref_bins, region)

    # Return the filtered chunks
    chunks = row_filter(
        convert(pixels, "polars"),
        pixels_filter(lower_id, upper_id),
        chunk_size,
    )

    return format_stream(chunks, df_dtype, as_chunks)


def subset_bin_region(
    bins: "DataFrame" | "DfStream",
    *,
    region: str,
    df_dtype: DfDtype = "polars",
    as_chunks: Bool = True,
    chunk_size: int = 10_000_000,
) -> "DataFrame" | "DfChunks":
    """Return a subset of bins within a genomic region."""

    def bins_filter(lower_id: int, upper_id: int) -> pl.Expr:
        """Filter to apply on the bins to find the subset."""
        return (pl.col("bin_id") >= lower_id) & (pl.col("bin_id") < upper_id)

    (bins,) = to_iterable(bins)
    lower_id, upper_id = find_extent(bins, region)

    # Return the filtered chunks
    chunks = row_filter(
        convert(bins, "polars"),
        bins_filter(lower_id, upper_id),
        chunk_size,
    )

    return format_stream(chunks, df_dtype, as_chunks)
