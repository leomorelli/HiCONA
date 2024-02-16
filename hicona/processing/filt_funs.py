"""Pixel table filtering functions to be applied through Scheduler objects.

This module contains functions which are loaded by the `FiltScheduler` class
constructor and are thus able to be loaded in the pre-filtering or the
post-filtering schedulers during pixels table normalization. For these reason
the functions are not really mean for direct usage.

All functions share a common interface/architecture:
- a pixel table object must always be provided as first argument
- other arguments might be present (but not always)
- an iterator of processed pixel chunks is returned
- defult arguments should be avoided (they can make filtering opaque)

# TODO: Maybe add check that a function can be applied at that step
"""

from ..chunked_ops import get_col_quantiles


def max_genomic_dist(table, max_dist):
    """
    Remove pixels whose genomic distance is higher than a threshold.
    Params:
        - max_dist: max allowed genomic distance among bins in bp.
    """

    for chunk in table.chunks():
        dist = (chunk.bin2_id - chunk.bin1_id) * table.binsize
        chunk = chunk.loc[dist <= max_dist]
        yield chunk


def norm_count_quant(table, quant):
    """
    Remove pixels whose normalized counts value is below a threshold.
    Params:
        - quant: remove pixels whose normalized value is below this quantile.
    """

    quant_val = get_col_quantiles(table.get_pixels(), quant)
    for chunk in table.chunks():
        chunk = chunk.loc[chunk.norm > quant_val]
        yield chunk


def self_looping_pix(table):
    """
    Remove self-looping pixels (pixels where bin1_id == bin2_id).
    Params:
        - None
    """

    for chunk in table.chunks():
        chunk = chunk.loc[chunk.bin1_id != chunk.bin2_id]
        yield chunk


def min_raw_counts(table, min_val):
    """
    Remove pixels whose raw count value is not greater than a threshold.
    Params:
        - min_val: remove pixels whose count is not greater than this value.
    """

    for chunk in table.chunks():
        chunk = chunk.loc[chunk.count > min_val]
        yield chunk


def rm_inter_chroms(table):
    """
    Remove inter chromosomal pixels.
    """
    pass
