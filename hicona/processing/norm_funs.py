import math

import cooler
import numpy as np
import pandas as pd

from ..utils.hdf5_ops import init_table, write_chunk


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


def add_bin_chroms(pix_iter, chr_bins, chr_names):
    """Placeholder"""

    bounds, names = get_chrom_binning(table.store, table.root)
    for chunk in pix_iter:
        for i in range(1, 3):
            bin_col, chr_col = f"bin{i}_id", f"bin{i}_chr"
            chunk[chr_col] = pd.cut(chunk[bin_col], chr_bins, labels=chr_names)
        yield chunk


def compute_curve(table):
    """Placeholder"""

    curve = pd.Series()

    for chunk in add_bin_chroms(table.get_chunks()):
        chunk["bin_diff"] = chunk.bin2_id - chunk.bin1_id
        vals = chunk.groupby(["bin_diff", "bin1_chr"]).count.apply(list)
        curve = curve.combine(vals, lambda x, y: x + y, fill_value=[])

    return curve.apply(median).rename("dist_norm")


class HiconaNorm:
    """Placeholder"""

    def get_norm_counts(self, table):
        """Iterator of table chunks normalized for genomic distance."""

        norm_curve = compute_curve(table)

        for chunk in add_bin_chroms(table.get_chunks()):
            chunk["bin_diff"] = chunk.bin2_id - chunk.bin1_id
            chunk = chunk.join(norm_curve, on=["bin_diff", "bin1_chr"])
            chunk["norm"] = np.log2(chunk["count"] / chunk["dist_norm"] + 1)
            yield chunk

    def apply(self, table):
        """Placeholder"""

        store = table.store
        root = table.pixels_uri
        table_len = None  # TODO: Get table lenght
        table_map = None  # TODO: Get table mapping

        init_table(store, root, table_len, table_map)

        lower = 0
        for chunk in self.get_norm_counts(table):
            write_chunk(store, root, chunk, lower, keys=["norm"])
            lower += len(chunk)


class PixNormManager:
    """Placeholder"""
     
    def __init__(self):
