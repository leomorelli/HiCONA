"""Placeholder"""

import math

import cooler
import numpy as np
import pandas as pd


CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX"]

# TODO: maybe make that adding chrom is default option in table iteration


def get_chrom_binning(cooler_uri):
    """Get bin boundaries of each chromosome.

    This is basically the same as fetching the chrom_offsets from cooler,
    but this way we avoid breaks due to the private interace changing.
    """

    handle = cooler.Cooler(cooler_uri)
    chrom_bins = [math.ceil(x) for x in handle.chromsizes[:] / handle.binsize]
    bin_bounds = np.cumsum([0] + chrom_bins)

    return bin_bounds, handle.chromnames


def chrom_binned_pixels(table):
    """Placeholder"""

    bins, names = get_chrom_binning(table.uris.cooler_uri())
    for chunk in table.chunks():
        for i in range(1, 3):
            bin_col, chr_col = f"bin{i}_id", f"bin{i}_chr"
            chunk[chr_col] = pd.cut(chunk[bin_col], bins, labels=names)
        yield chunk


# Substitute with cooler annotate?
