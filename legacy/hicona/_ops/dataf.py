"""Placeholder"""

import numpy as np
import pandas as pd
import scipy as sp
import polars as pl


def pd2gt_dtype(pd_dtype: str):
    """Convert pandas-like datatypes to graph-tools datatypes"""

    conv_dict = {
        "int32": "int",
        "int64": "long",
        "float64": "long double",
        "object": "string",
        "bool": "bool",
        "category": "string",
    }

    return conv_dict[pd_dtype]


def swap_columns(dataf: pd.DataFrame, col1: str, col2: str):
    """Swap the columns to make the row sorted alphabetically."""

    dataf[col1], dataf[col2] = np.where(
        (dataf[col1] < dataf[col2]),
        (dataf[col1], dataf[col2]),
        (dataf[col2], dataf[col1]),
    )


def odds_ratios(
    kept: pd.Series,
    full: pd.Series,
    haldane: bool = False,
) -> tuple[list[float], list[float]]:
    """Compute log-odds ratios and p-values for a modalities vector.

    `kept` and `full` are two vectors of the same length representing two
    groups, one of which is a strict superset of the other. Corresponding
    elements in the two vectors represent the counts of the same modality in
    the two groups. The function computes the log-odds ratios and p-values for
    each modality.

    It is assumed that the vector represents all modalities of the group and
    thus the sum of the vector is equal to group cardinality.
    """

    # Compute the second column of the contingency table
    disc = full - kept

    # If either group is empty, return NaNs
    if all(kept == 0) or all(disc == 0):
        return ([np.nan] * len(kept), [np.nan] * len(kept))

    # Lower marginals of the contingency table
    kept_tot = kept.sum()
    disc_tot = disc.sum()

    # Containers for the results
    odds_vect = []
    pval_vect = []

    # Compute the log-odds ratios and p-values for each annotation
    for kept_ann, disc_ann in zip(kept, disc):

        # Bottom left and bottom right cells of the contingency table
        kept_other = kept_tot - kept_ann
        disc_other = disc_tot - disc_ann

        # Create the full contingency table
        table = np.array([[kept_ann, disc_ann], [kept_other, disc_other]])

        # Compute the p-value on the table without any shift
        pval_vect.append(sp.stats.fisher_exact(table).pvalue)  # type: ignore

        # Apply Haldane's correction if needed
        if haldane:
            table = table + 0.5

        # Compute the log-odds ratio
        odds = table[0, 0] * table[1, 1] / (table[0, 1] * table[1, 0])
        odds_vect.append(np.log2(odds))

    return odds_vect, pval_vect


def add_percentiles(df: pl.DataFrame, col_name: str):
    """Add a column with the percentiles of the values in the given column.

    In case the same value is assigned to multiple percentiles, select
    the highest percentile. This should not have any impact if the
    percentiles are used for ranking purposes.
    """

    quants = np.linspace(0, 1, 101)
    values = [df.select(pl.col(col_name).quantile(q)).item() for q in quants]
    cut_df = (
        pl.DataFrame({"quant": quants, "value": values})
        .group_by("value")
        .max()
        .with_columns((pl.col("quant") * 100).cast(int))
        .sort("value")
    )

    df = df.with_columns(
        (
            pl.col(col_name)
            .cut(
                cut_df.select("value").to_numpy()[1:],  # type: ignore
                labels=cut_df.select("quant").cast(pl.String).to_series().to_list(),
                left_closed=True,
            )
            .cast(pl.Int32)
        ).alias("quant")
    )

    return df
