"""Placeholder"""

import statistics as stat

import cooler
import numpy as np
import pandas as pd

from .auxiliary_funs import chrom_binned_pixels


def hicona_norm(table, apply_col):
    """Placeholder"""

    def get_norm_curve(table):
        """"""

        curve = pd.DataFrame()

        for chunk in chrom_binned_pixels(table):
            print(chunk)
            chunk["diff"] = chunk.bin2_id - chunk.bin1_id
            parts = chunk.groupby(["diff", "bin1_chr"], as_index=False)["count"].apply(
                list
            )
            print(parts)
            curve = curve.combine(parts, lambda x, y: x + y, fill_value=[])

        return curve.apply(stat.median).rename("dist_norm")

    norm_curve = get_norm_curve(table)
    print(norm_curve)

    for chunk in chrom_binned_pixels(table):
        chunk["diff"] = chunk.bin2_id - chunk.bin1_id
        chunk = chunk.merge(norm_curve, on=["diff", "bin1_chr"])
        yield np.log2(chunk[apply_col] / chunk["dist_norm"] + 1)

        # TODO: Probably need to convert to pandas series named norm


def no_norm(table):
    """Placeholder"""

    for chunk in table.chunks():
        # TODO: yield filt table chunks

        pass


def ice_norm(table, colname):
    """Placeholder"""

    handle = cooler.Cooler(table.cooler_uri())
    norm_vector = handle.bins()[:][colname]

    for chunk in table.chunks():
        pass
