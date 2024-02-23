"""Pixel table normalization functions to be applied through a Scheduler.

This module contains functions which are loaded by the `NormScheduler` class
constructor and are thus able to be loaded in the normalization scheduler. 
For these reason the functions are not really mean for direct usage.

All functions share a common interface/architecture:
- a pixel table object must always be provided as first argument
- other arguments might be present (but not always)
- an iterator of processed pixel chunks is returned
- defult arguments should be avoided (they can make normalization opaque)

# TODO: Ideally multiple normalization functions should be able to be applied,
# but the current implementation does not allow for that yet.
"""

import statistics as stat

import cooler
import numpy as np
import pandas as pd

from .auxiliary_funs import chrom_binned_pixels


def hicona_norm(table, apply_col):
    """
    Apply default HiCONA normalization to a table (genomic distance).
    Params:
        - apply_col: column to apply the normalization to.
    """

    def get_norm_curve(table):
        """Compute the normalization curve for a table.

        The normalization curve is the median of the counts for each genomic
        distance, for each chromosome, in the table.
        """

        curve = pd.Series()

        for chunk in chrom_binned_pixels(table):
            chunk["diff"] = chunk.bin2_id - chunk.bin1_id
            grp_cols = ["diff", "bin1_chr"]
            part = chunk.groupby(grp_cols, observed=True)["count"].apply(list)
            curve = curve.combine(part, lambda x, y: x + y, fill_value=[])

        curve = curve.rename_axis(index=["diff", "bin1_chr"])
        return curve.apply(stat.median).rename("dist_norm")

    norm_curve = get_norm_curve(table)

    for chunk in chrom_binned_pixels(table):
        chunk["diff"] = chunk.bin2_id - chunk.bin1_id
        chunk = chunk.merge(norm_curve, how="left", on=["diff", "bin1_chr"])
        norm_col = np.log2(chunk[apply_col] / chunk["dist_norm"] + 1)
        yield pd.DataFrame({"norm": norm_col})


def no_norm(table):
    """
    Do not apply any normalization, just copy raw values to norm column.
    Params:
        - None
    """

    for chunk in table.chunks():
        # TODO: yield filt table chunks

        pass


def ice_norm(table, colname, apply_col):
    """
    Apply ICE, or any bin-wise, multiplicative normalization, to a table.
    Params:
        - colname: bin table column to use as the normalization vector.
        - apply_col: column to apply the normalization to.
    """

    handle = cooler.Cooler(table.cooler_uri())
    norm_vector = handle.bins()[:][colname]

    for chunk in table.chunks():
        pass
