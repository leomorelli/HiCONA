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

import numpy as np
import pandas as pd

from .auxiliary_funs import chrom_binned_pixels
from ..hicona_table import RawTable
from ..utils.chunked_ops import groupwise_median


def distance_norm(table: RawTable, apply_col: str):
    """
    Apply default HiCONA normalization to a table (genomic distance).
    Params:
        - apply_col: column to apply the normalization to.
    """

    def chunks_with_distance(table):
        for chunk in chrom_binned_pixels(table):
            chunk["diff"] = chunk.bin2_id - chunk.bin1_id
            yield chunk

    grouping_cols = ["diff", "bin1_chr"]
    norm_curve = groupwise_median(chunks_with_distance(table), grouping_cols, apply_col)

    for chunk in chunks_with_distance(table):
        chunk = chunk.merge(norm_curve, how="left", on=grouping_cols)
        norm_col = np.log2(chunk[apply_col] / chunk["dist_norm"] + 1)
        yield pd.DataFrame({"norm": norm_col})


def no_norm(table: RawTable):
    """
    Do not apply any normalization, just copy raw values to norm column.
    Params:
        - None
    """

    for chunk in table.chunks():
        yield pd.DataFrame({"norm": chunk["count"]})


def binwise_norm(
    table: RawTable,
    apply_col: str,
    colname: str,
    divisive: bool = False,
    drop_nas: bool = False,
):
    """
    Apply a bin-wise normalization (one norm factor per each bin) to a table.
    Params:
        - colname: bin table column to use as the normalization vector.
        - apply_col: column to apply the normalization to.
        - divisive: whether to divide or multiply by the norm factor.
    """

    # NOTE: This function mimicks matrix balancing normalization in cooler.

    col1, col2 = f"{colname}1", f"{colname}2"

    for chunk in table.chunks(annotated=True):
        if divisive:
            chunk[col1] = 1 / chunk[col1]
            chunk[col2] = 1 / chunk[col2]
        chunk["norm"] = chunk[apply_col] * chunk[col1] * chunk[col2]

        if drop_nas:
            chunk.dropna(inplace=True)
        else:
            chunk = chunk[["norm"]]

        yield chunk
