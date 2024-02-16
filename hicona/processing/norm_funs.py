"""Placeholder"""

import math

import cooler
import numpy as np
import pandas as pd


CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX"]

# TODO: maybe make that adding chrom is default option in table iteration


def get_chrom_binning(store, root):
    """Get bin boundaries of each chromosome.

    This is basically the same as fetching the chrom_offsets from cooler,
    but this way we avoid breaks due to the private interace changing.
    """

    handle = cooler.Cooler("::".join([store, root]))
    chrom_bins = [math.ceil(x) for x in handle.chromsizes[:] / handle.binsize]
    bin_bounds = np.cumsum([0] + chrom_bins)

    return bin_bounds, handle.chromnames


def get_column_binning(iterator):
    """Placeholder."""

    chr_bins, chr_names = get_chrom_binning(table.store, table.root)


def hicona_norm(table, apply_col):
    """Placeholder"""

    def get_norm_curve(table):
        """"""

        curve = pd.Series()

        for chunk in table.get_chunks():
            # S
            chunk["bin_diff"] = chunk.bin2_id - chunk.bin1_id
            chunk["chrom"] = pd.cut(chunk.bin1_id, chr_bins, labels=chr_names)

            parts = chunk.groupby(["bin_diff", "chrom"]).count.apply(list)
            curve = curve.combine(parts, lambda x, y: x + y, fill_value=[])

        return curve.apply(median).rename("dist_norm")

    norm_curve = get_norm_curve(table)

    for chunk in add_bin_chroms(table.get_chunks()):
        chunk["bin_diff"] = chunk.bin2_id - chunk.bin1_id
        chunk = chunk.join(norm_curve, on=["bin_diff", "bin1_chr"])
        yield np.log2(chunk[apply_col] / chunk["dist_norm"] + 1)

        # TODO: Probably need to convert to pandas series named norm


def no_norm(table):
    """Placeholder"""

    for chunk in table.get_chunks():
        # TODO: yield filt table chunks

        pass
