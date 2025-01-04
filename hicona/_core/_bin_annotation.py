"""Module with functions for bin annotation."""

from typing import Literal, TYPE_CHECKING

import bioframe as bf
import polars as pl
import pandas as pd


if TYPE_CHECKING:
    from .._utils.df_dtypes import DataFrame

__all__ = ("get_annotated_bins",)

BASE_BIN_COLS: tuple[str, str, str] = ("chrom", "start", "end")
MAX_MODS_COLS: int = 15

AnnoMetric = Literal["bp_overlap", "frac_overlap", "chrom_enrich"]


# TODO: there still is a lot of hard-coded column names which is not great. Refactor.
# Maybe use an Enum

########################################################################################
# Auxiliary functions
########################################################################################


def bp_to_enrichment(df: pl.DataFrame) -> pl.DataFrame:
    """Use the most enriched bin annotation to select the row.

    The procedure is as follows:

        - for each chromosome compute the fraction of bases for each annotation.
        - for each bin compute the fraction of bases for each annotation.
        - compute the fold change between the bin fraction over chrom fraction for
          each annotation.
        - compute the log2 of the fold change + 1
        - for each bin only keep the most enriched annotation
        - if there are annotations with the same enrichment, keep the one occupying
          the higher fraction of bases in the bin (if still tied, first found).
    """

    # It assumed that the table has only 5 columns (base ones, bp_overlap and annotation)
    cols: list[str] = [c for c in df.columns if c not in BASE_BIN_COLS]
    cols.remove("bp_overlap")
    annot_column: str = cols.pop()
    if cols:
        raise RuntimeError("Table had more columns than expected.")

    # In order to be able to perform the joins, there can be no nulls in the annotation
    df = df.fill_null("None")

    # At this point you have a df with multiple intersections per bin (one per modality).
    # Summing over all intersections should give the entire chromosome size assuming
    # the annotation covers all bases (maybe the annotation does not by default, but the
    # overlap function should introduce `None` modality.)

    anno_sizes: pl.DataFrame = df.group_by(("chrom", annot_column)).agg(
        pl.sum("bp_overlap").alias("anno_size")
    )
    chrom_sizes: pl.DataFrame = anno_sizes.group_by("chrom").agg(
        pl.sum("anno_size").alias("chrom_size")
    )

    # Background to use for the enrichment
    chrom_fracs = (
        anno_sizes.join(chrom_sizes, on="chrom", how="left")
        .with_columns((pl.col("anno_size") / pl.col("chrom_size")).alias("chrom_frac"))
        .select(["chrom", annot_column, "chrom_frac"])
    )

    return (
        df.join(chrom_fracs, on=["chrom", annot_column], how="left")
        .with_columns(
            (
                pl.col("bp_overlap")
                / (pl.col("end") - pl.col("start"))  # Normalize by bin size
                / pl.col("chrom_frac")  # Normalize by fraction in chromosome
            )
            .add(1)
            .log(base=2)
            .alias("chrom_enrich")
        )
        .drop("bp_overlap", "chrom_frac")
    )


def bp_to_fraction(df: pl.DataFrame) -> pl.DataFrame:
    """Convert overlap from absolute bp number to fraction of bin size."""
    return df.with_columns(
        (pl.col("bp_overlap") / (pl.col("end") - pl.col("start")))
    ).rename({"bp_overlap": "frac_overlap"})


def overlap_with_size(df_a: "DataFrame", df_b: "DataFrame") -> pl.DataFrame:
    """Overlap function from bioframe with added overlap_size column."""

    df_a = df_a if isinstance(df_a, pd.DataFrame) else df_a.to_pandas()
    df_b = df_b if isinstance(df_b, pd.DataFrame) else df_b.to_pandas()
    annot_col: str = [c for c in df_b.columns if c not in BASE_BIN_COLS].pop()

    merged_df: pl.DataFrame = (
        pl.from_pandas(bf.overlap(df_a, df_b, return_overlap=True))
        .with_columns(bp_overlap=(pl.col("overlap_end") - pl.col("overlap_start")))
        .drop(["chrom_", "start_", "end_", "overlap_start", "overlap_end"])
        .rename(lambda c: c.rstrip("_"))
    )

    # After the intersection, there will be rows with annotation null while
    # others with no null but not all bases of the interval are annotated.
    # In either cases, the rows will not sum to the bin size, therefore we
    # need to compute the size of the interval, subtract the sum of the sizes
    # of all annotations for the interval, and add back this as null annotation.
    # NOTE: The null annotation is therefore always consolidated.

    null_rows: pl.DataFrame = (
        merged_df.group_by(BASE_BIN_COLS)
        .agg(pl.sum("bp_overlap"))
        .with_columns(bp_overlap=pl.col("end") - pl.col("start") - pl.col("bp_overlap"))
        .filter(pl.col("bp_overlap") != pl.lit(0))
        .with_columns(pl.lit(None).alias(annot_col))
        .select([*BASE_BIN_COLS, annot_col, "bp_overlap"])
    )

    return (
        merged_df.filter(pl.col("bp_overlap").is_not_null())
        .vstack(null_rows)
        .sort(BASE_BIN_COLS)
    )


########################################################################################
########################################################################################


def get_annotated_bins(
    bins_df: "DataFrame",
    anno_df: "DataFrame",
    metric: AnnoMetric,
    consolidate: bool,
    save_all_mods: bool,
) -> pl.DataFrame:
    """Return a bin table annotated according to a second dataframe.

    Strategy refers to the function to use to resolve duplicate rows, since in a
    bin table each bin can have at most one annotation per column.
    """

    # Bad parameter combinations
    if metric == "chrom_enrich" and not consolidate:
        raise ValueError("Cannot use metric `chrom_enrich` if `consolidate = False`")
    if save_all_mods and not consolidate:
        raise ValueError("Cannot save all modalities if `consolidate = False`")

    # Ensure that initially you are working with pandas dataframes.
    bins_df = bins_df if isinstance(bins_df, pd.DataFrame) else bins_df.to_pandas()
    anno_df = anno_df if isinstance(anno_df, pd.DataFrame) else anno_df.to_pandas()
    annot_col: str = [c for c in anno_df.columns if c not in BASE_BIN_COLS].pop()

    # Check to prevent exploding the number of columns in the file due to
    # a categorical-like variable with too many modalities.
    if save_all_mods:
        num_mods: int = anno_df[annot_col].nunique()
        if num_mods > MAX_MODS_COLS:
            raise ValueError(f"Trying to generate too many columns: {num_mods}")

    # Perform the initial merge; this table can have multiple intersections
    merge_df: pl.DataFrame = overlap_with_size(bins_df, anno_df)
    if consolidate:
        merge_df = merge_df.group_by(anno_df.columns, maintain_order=True).agg(
            pl.sum("bp_overlap")
        )

    # Convert the bp overlap into the metric of interest
    # Multiple intersections can still be present at this point
    # NOTE: These operations can be seen as sequential, but they are treated as distinct
    # cases since a new metric might be added which is not sequential to these ones.
    match metric:
        case "bp_overlap":
            pass
        case "chrom_enrich":
            merge_df = bp_to_enrichment(merge_df)
        case "frac_overlap":
            merge_df = bp_to_fraction(merge_df)
        case _:
            raise ValueError(f"{metric} is not a valid annotation metric.")

    # Handle the multiple intersection issue by either returning them all in a
    # split manner or by taking the one with the highest metric for each bin.
    if save_all_mods:
        return merge_df.pivot(index=BASE_BIN_COLS, on=annot_col).rename(
            lambda col: f"{col}_{metric}" if col not in BASE_BIN_COLS else col
        )

    return merge_df.group_by(BASE_BIN_COLS, maintain_order=True).agg(
        pl.col(annot_col).get(pl.col(metric).arg_max())
    )
