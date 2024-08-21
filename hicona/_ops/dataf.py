"""Placeholder"""

import numpy as np
import pandas as pd
import scipy as sp
import polars as pl

# TODO: somehow add category modality


def from_df_to_sarrays(data: pd.DataFrame):
    """Return each column of a dataframe as a numpy structured array.

    Transform the columns of a dataframe into numpy structured array and
    define the numpy datatype most appropriate for storage in HDF5 (especially
    minimum required string fixed length for categorical annotations).
    """

    # TODO: Currently only discriminating string/non string, improve

    dtypes = data.dtypes
    for col_name, dtype in zip(data, dtypes):
        if dtype == "object":
            str_len = int(data[col_name].str.len().max())
            str_len = str_len if str_len > 3 else 3
            dtype = np.dtype(f"S{str_len}")

        new_col = np.zeros(len(data), dtype)
        new_col[:] = data[col_name].values

        yield (col_name, new_col, dtype)


def pd_to_h5_dtype(pd_dtype: str):
    """Placeholder"""

    h5_dtype = pd_dtype
    if pd_dtype == "object":
        h5_dtype = np.dtype("S10")

    return h5_dtype


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
