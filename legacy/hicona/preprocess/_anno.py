"""Annotation functions for Hi-C data."""

# TODO: Maybe make into new class of operations?

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from hicona._core import PixelTable
    from hicona._dtypes import DfChunks


def add_annot(table: "PixelTable", col: str) -> "DfChunks":
    """Get a dataframe with the columns `bin_id` and `chrom` for join purposes."""

    chrom_ann = (
        table.bins.dataframe()
        .select(col)
        .with_row_index("bin_id")
        .with_columns(pl.col("bin_id").cast(pl.Int64))
        .collect()
    )

    # TODO: CategoricalRemappingWarning is raised in this area, fix
    for chunk in table.chunks():
        yield (
            chunk.join(chrom_ann, left_on="bin1_id", right_on="bin_id", how="left")
            .join(chrom_ann, left_on="bin2_id", right_on="bin_id", how="left")
            .rename({col: f"{col}1", f"{col}_right": f"{col}2"})
        )


def add_genomic_dist(table: "PixelTable") -> "DfChunks":
    """Add genomic distance column to a pixel table."""

    for chunk in add_annot(table, "chrom"):

        yield chunk.with_columns(
            ((pl.col("bin2_id") - pl.col("bin1_id")) * table.bin_size).alias("dist")
        )
